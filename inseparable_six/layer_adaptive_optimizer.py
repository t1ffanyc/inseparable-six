"""
Layer-Adaptive Differential Privacy Optimizer

Implements per-layer adaptive clipping bounds and noise scaling for DP-SGD.
Integrates with allocation strategies from allocation_strategies.py.

Key concepts:
- global_clipping_bound (C_global): Overall gradient clipping bound
- layer_clipping_bound (C_l): Per-layer clipping bound where sqrt(sum(C_l^2)) = C_global
- noise_multiplier (σ): Controls privacy-utility tradeoff
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import (
    _check_processed_flag, 
    _mark_as_processed, 
    _generate_noise
)
from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator
from tqdm import tqdm

from .allocation_strategies import AllocationStrategy, UniformStrategy
from .metrics import MetricsCollector, AllocationMetrics, ExperimentResult
from .config import ExperimentConfig, GradSampleMode


@dataclass
class LayerClippingConfig:
    """Configuration for a single parameter group's clipping"""
    group_name: str
    parameters: List[nn.Parameter]
    clipping_bound: float  # C_l for this group


class LayerAdaptiveDPOptimizer(DPOptimizer):
    """
    DPOptimizer with per-layer adaptive clipping and noise.
    
    Instead of uniform clipping across all parameters, this optimizer:
    1. Clips each parameter group to its assigned bound C_l
    2. Adds noise scaled proportionally to each C_l
    
    Privacy guarantee: Maintained because sqrt(sum(C_l^2)) = global_clipping_bound,
    ensuring the total sensitivity equals the global bound.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        global_clipping_bound: float,
        layer_configs: List[LayerClippingConfig],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs
    ):
        """
        Args:
            optimizer: Base optimizer to wrap
            noise_multiplier: Noise scale σ (same for all layers for privacy accounting)
            global_clipping_bound: Global clipping bound C_global
            layer_configs: Per-layer clipping configurations
            expected_batch_size: Expected batch size for gradient averaging
            loss_reduction: How loss is reduced ("mean" or "sum")
            generator: Random number generator for noise
            secure_mode: Use cryptographically secure RNG
        """
        self.layer_configs = layer_configs
        self.global_clipping_bound = global_clipping_bound
        
        # Verify constraint
        self._verify_clipping_constraint()
        
        # Initialize parent
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=global_clipping_bound,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs
        )
        
        # Build parameter -> clipping bound mapping
        self._build_param_mapping()
    
    def _verify_clipping_constraint(self):
        """Verify that sqrt(sum(C_l^2)) = global_clipping_bound"""
        sum_sq = sum(cfg.clipping_bound ** 2 for cfg in self.layer_configs)
        computed = np.sqrt(sum_sq)
        
        if not np.isclose(computed, self.global_clipping_bound, rtol=1e-3):
            raise ValueError(
                f"Clipping constraint violated: "
                f"sqrt(sum(C_l^2)) = {computed:.6f} != {self.global_clipping_bound:.6f}"
            )
    
    def _build_param_mapping(self):
        """Build mapping from parameter to its clipping bound"""
        self.param_to_config: Dict[nn.Parameter, LayerClippingConfig] = {}
        
        for config in self.layer_configs:
            for param in config.parameters:
                if param.requires_grad:
                    self.param_to_config[param] = config
        
        # Verify all optimizer params have mappings
        missing = [p for p in self.params if p not in self.param_to_config]
        if missing:
            raise ValueError(
                f"{len(missing)} trainable parameters have no clipping config"
            )
    
    def clip_and_accumulate(self):
        """
        Per-layer gradient clipping.
        
        For each parameter with bound C_l:
        1. Compute per-sample gradient norms
        2. Clip to C_l
        3. Aggregate clipped gradients
        """
        if len(self.grad_samples[0]) == 0:
            return
        
        for param in self.params:
            _check_processed_flag(param.grad_sample)
            
            config = self.param_to_config.get(param)
            if config is None:
                raise ValueError("Parameter missing clipping configuration")
            
            clipping_bound = config.clipping_bound
            
            # Get per-sample gradients
            grad_sample = self._get_flat_grad_sample(param)
            
            # Compute per-sample norms
            per_sample_norms = grad_sample.reshape(len(grad_sample), -1).norm(2, dim=1)
            
            # Clip: factor = min(1, C_l / norm)
            clip_factor = (clipping_bound / (per_sample_norms + 1e-6)).clamp(max=1.0)
            
            # Apply clipping and sum across batch
            clipped_grad = torch.einsum("i,i...", clip_factor, grad_sample)
            
            # Accumulate
            if param.summed_grad is not None:
                param.summed_grad += clipped_grad
            else:
                param.summed_grad = clipped_grad
            
            _mark_as_processed(param.grad_sample)
    
    def add_noise(self):
        """
        Add per-layer scaled Gaussian noise.
        
        For parameter with bound C_l, noise ~ N(0, (σ * C_l)^2)
        """
        for param in self.params:
            if param.summed_grad is None:
                continue
            
            _check_processed_flag(param.summed_grad)
            
            config = self.param_to_config.get(param)
            if config is None:
                raise ValueError("Parameter missing clipping configuration")
            
            # Noise std = noise_multiplier * C_l
            noise_std = self.noise_multiplier * config.clipping_bound
            
            noise = _generate_noise(
                std=noise_std,
                reference=param.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            
            param.grad = (param.summed_grad + noise).view_as(param)
            _mark_as_processed(param.summed_grad)
    
    def get_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about layer clipping bounds.
        
        Returns:
            Dictionary mapping group names to their stats
        """
        stats = {}
        for config in self.layer_configs:
            stats[config.group_name] = {
                "clipping_bound": config.clipping_bound,
                "num_parameters": len(config.parameters),
                "total_params": sum(p.numel() for p in config.parameters),
            }
        return stats


# =============================================================================
# Setup Utilities
# =============================================================================

def create_layer_configs(
    model: nn.Module,
    allocation_strategy: AllocationStrategy,
    global_clipping_bound: float,
    data_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[List[LayerClippingConfig], AllocationMetrics]:
    """
    Create layer clipping configurations using an allocation strategy.
    
    Args:
        model: Model (potentially with LoRA adapters)
        allocation_strategy: Strategy for computing per-layer bounds
        global_clipping_bound: Global clipping bound C_global
        data_loader: Optional data for gradient-based strategies
        device: Device for computations
        verbose: Print progress information
        
    Returns:
        Tuple of (layer_configs, allocation_metrics)
    """
    if verbose:
        print(f"\n[Computing Layer Allocations]")
        print(f"  Strategy: {allocation_strategy.name}")
        print(f"  Global clipping bound: {global_clipping_bound}")
    
    # Compute allocations
    clipping_bounds = allocation_strategy.compute_allocation(
        model=model,
        global_clipping_bound=global_clipping_bound,
        data_loader=data_loader,
        device=device,
    )
    
    # Get parameter groups
    param_groups = allocation_strategy.get_parameter_groups(model)
    
    # Create layer configs
    layer_configs = []
    for group_name, params in param_groups.items():
        if group_name not in clipping_bounds:
            raise ValueError(f"No allocation for group: {group_name}")
        
        layer_configs.append(LayerClippingConfig(
            group_name=group_name,
            parameters=params,
            clipping_bound=clipping_bounds[group_name],
        ))
    
    # Get metrics from strategy
    allocation_metrics = allocation_strategy.get_metrics()
    
    if verbose:
        print(f"  ✓ Created {len(layer_configs)} layer configurations")
        bounds = [cfg.clipping_bound for cfg in layer_configs]
        print(f"  ✓ C_l range: [{min(bounds):.4f}, {max(bounds):.4f}]")
        
        # Verify constraint
        sum_sq = sum(b**2 for b in bounds)
        print(f"  ✓ Constraint: sqrt(sum(C_l^2)) = {np.sqrt(sum_sq):.6f}")
    
    return layer_configs, allocation_metrics


def make_layer_adaptive_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    allocation_strategy: AllocationStrategy,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    global_clipping_bound: float,
    estimation_data_loader: Optional[DataLoader] = None,
    batch_first: bool = True,
    loss_reduction: str = "mean",
    poisson_sampling: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[nn.Module, LayerAdaptiveDPOptimizer, DataLoader, PrivacyEngine, AllocationMetrics]:
    """
    Set up layer-adaptive DP training targeting a specific (ε, δ) budget.
    
    This is the main entry point for layer-adaptive DP. It:
    1. Computes per-layer allocations using the strategy
    2. Wraps the model for per-sample gradients
    3. Computes noise multiplier for target privacy
    4. Creates the layer-adaptive optimizer
    5. Prepares the data loader with privacy sampling
    
    Args:
        model: Model to make private
        optimizer: Base optimizer
        data_loader: Training data loader
        allocation_strategy: Strategy for computing per-layer bounds
        target_epsilon: Target privacy budget ε
        target_delta: Target δ
        epochs: Number of training epochs
        global_clipping_bound: Global clipping bound C_global
        estimation_data_loader: Separate data for gradient estimation (optional)
        batch_first: Whether batch dimension is first in tensors
        loss_reduction: "sum" or "mean"
        poisson_sampling: Use Poisson sampling for privacy
        device: Device for computations
        verbose: Print setup progress
        
    Returns:
        Tuple of:
        - wrapped_model: Model with GradSampleModule
        - layer_optimizer: LayerAdaptiveDPOptimizer
        - dp_data_loader: Privacy-aware data loader
        - privacy_engine: PrivacyEngine for tracking privacy spent
        - allocation_metrics: Metrics from allocation computation
    """
    if verbose:
        print("\n" + "=" * 70)
        print("LAYER-ADAPTIVE DP SETUP")
        print("=" * 70)
    
    privacy_engine = PrivacyEngine()
    
    # Step 1: Validate model
    if verbose:
        print(f"\n[1/5] Validating model...")
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        if verbose:
            print(f"  Fixing {len(errors)} incompatible modules...")
        model = ModuleValidator.fix(model)
    else:
        if verbose:
            print("  ✓ Model is Opacus-compatible")
    
    # Step 2: Compute layer allocations
    if verbose:
        print(f"\n[2/5] Computing layer allocations...")
    layer_configs, allocation_metrics = create_layer_configs(
        model=model,
        allocation_strategy=allocation_strategy,
        global_clipping_bound=global_clipping_bound,
        data_loader=estimation_data_loader or data_loader,
        device=device,
        verbose=verbose,
    )
    
    # Step 3: Wrap model
    if verbose:
        print(f"\n[3/5] Wrapping model with GradSampleModule...")
    if not isinstance(model, GradSampleModule):
        wrapped_model = GradSampleModule(
            model,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )
    else:
        wrapped_model = model
    if verbose:
        print("  ✓ Model wrapped")
    
    # Step 4: Compute noise multiplier
    if verbose:
        print(f"\n[4/5] Computing noise multiplier for ε={target_epsilon}, δ={target_delta:.2e}...")
    
    batch_size = data_loader.batch_size if hasattr(data_loader, 'batch_size') else 1
    sample_rate = batch_size / len(data_loader.dataset)
    
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant="rdp",
    )
    
    if verbose:
        print(f"  ✓ Noise multiplier (σ): {noise_multiplier:.4f}")
        print(f"  ✓ Sample rate: {sample_rate:.6f}")
    
    # Step 5: Create optimizer
    if verbose:
        print(f"\n[5/5] Creating LayerAdaptiveDPOptimizer...")
    
    layer_optimizer = LayerAdaptiveDPOptimizer(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        global_clipping_bound=global_clipping_bound,
        layer_configs=layer_configs,
        expected_batch_size=batch_size,
        loss_reduction=loss_reduction,
        generator=None,
        secure_mode=False,
    )
    
    if verbose:
        print(f"  ✓ Optimizer created with {len(layer_configs)} layer groups")
    
    # Prepare data loader
    if poisson_sampling:
        dp_data_loader = DPDataLoader.from_data_loader(
            data_loader,
            distributed=False,
        )
    else:
        dp_data_loader = data_loader
    
    # Set up privacy accountant
    accountant = RDPAccountant()
    layer_optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
    )
    privacy_engine.accountant = accountant
    
    if verbose:
        print("\n" + "=" * 70)
        print("Layer-Adaptive DP setup complete!")
        print("=" * 70 + "\n")
    
    return (
        wrapped_model, 
        layer_optimizer, 
        dp_data_loader, 
        privacy_engine, 
        allocation_metrics
    )


# =============================================================================
# Layer-Adaptive Trainer
# =============================================================================

class LayerAdaptiveDPTrainer:
    """
    Trainer for layer-adaptive DP with comprehensive metrics collection.
    
    Extends baseline DP training with per-layer adaptive clipping.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        allocation_strategy: AllocationStrategy,
        estimation_data_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            config: Experiment configuration
            allocation_strategy: Strategy for per-layer allocation
            estimation_data_loader: Optional separate data for gradient estimation
        """
        self.config = config
        self.allocation_strategy = allocation_strategy
        self.estimation_data_loader = estimation_data_loader
        
        # Components (initialized in setup())
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[LayerAdaptiveDPOptimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.allocation_metrics: Optional[AllocationMetrics] = None
        
        # Metrics
        self.metrics_collector = MetricsCollector(
            f"{config.name}_{allocation_strategy.name}"
        )
        
        self.is_setup = False
        
        # Set seed
        self._set_seed()
    
    def _set_seed(self):
        """Set random seeds"""
        seed = self.config.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> "LayerAdaptiveDPTrainer":
        """
        Set up training with provided model and data.
        
        Args:
            model: Model to train (already prepared with LoRA if desired)
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Self for method chaining
        """
        self.test_loader = test_loader
        
        # Set delta if needed
        if self.config.privacy.target_delta is None:
            self.config.privacy.target_delta = 1.0 / len(train_loader.dataset)
        
        # Create base optimizer
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Set up layer-adaptive DP
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.privacy_engine,
            self.allocation_metrics,
        ) = make_layer_adaptive_private(
            model=model,
            optimizer=base_optimizer,
            data_loader=train_loader,
            allocation_strategy=self.allocation_strategy,
            target_epsilon=self.config.privacy.target_epsilon,
            target_delta=self.config.privacy.target_delta,
            epochs=self.config.training.epochs,
            global_clipping_bound=self.config.privacy.max_grad_norm,
            estimation_data_loader=self.estimation_data_loader,
            device=self.config.device,
        )
        
        # Store noise multiplier in config
        self.config.privacy.noise_multiplier = self.optimizer.noise_multiplier
        
        # Set up metrics
        self.metrics_collector.set_config(self.config.to_dict())
        self.metrics_collector.set_model_info(
            model_name=self.config.model.name,
            total_params=sum(p.numel() for p in self.model.parameters()),
            trainable_params=sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            lora_enabled=self.config.lora.enabled,
        )
        self.metrics_collector.set_allocation_metrics(self.allocation_metrics)
        
        self.is_setup = True
        return self
    
    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Train for one epoch"""
        if not self.is_setup:
            raise RuntimeError("Call setup() before training")
        
        self.model.train()
        self.metrics_collector.start_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
        )
        
        for step, batch in enumerate(progress):
            step_start = time.time()
            
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch.get("labels", batch.get("label")).to(self.config.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            loss_val = loss.item()
            total_loss += loss_val
            preds = torch.argmax(outputs.logits, dim=-1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            step_time = (time.time() - step_start) * 1000
            
            self.metrics_collector.log_step(
                step=step,
                epoch=epoch,
                loss=loss_val,
                accuracy=100 * batch_correct / labels.size(0),
                batch_size=labels.size(0),
                learning_rate=self.config.training.learning_rate,
                step_time_ms=step_time,
            )
            
            progress.set_postfix({
                'loss': f"{loss_val:.4f}",
                'acc': f"{100 * correct / total:.2f}%",
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        try:
            epsilon = self.privacy_engine.accountant.get_epsilon(
                delta=self.config.privacy.target_delta
            )
        except Exception:
            epsilon = float('nan')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        }
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch.get("labels", batch.get("label")).to(self.config.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(self.test_loader), 100 * correct / total
    
    def train(self) -> ExperimentResult:
        """Full training loop"""
        if not self.is_setup:
            raise RuntimeError("Call setup() before training")
        
        print(f"\nStarting layer-adaptive training...")
        print(f"Strategy: {self.allocation_strategy.name}\n")
        
        self.metrics_collector.start_experiment()
        
        for epoch in range(self.config.training.epochs):
            metrics = self.train_epoch(epoch)
            eval_loss, eval_acc = self.evaluate()
            
            self.metrics_collector.end_epoch(
                train_loss=metrics['loss'],
                train_accuracy=metrics['accuracy'],
                eval_accuracy=eval_acc,
                eval_loss=eval_loss,
                epsilon=metrics['epsilon'],
                delta=self.config.privacy.target_delta,
            )
            
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Train Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%")
            print(f"  Eval Acc: {eval_acc:.2f}%")
            print(f"  Epsilon: {metrics['epsilon']:.2f}" if not (metrics['epsilon'] != metrics['epsilon']) else "  Epsilon: N/A")
        
        try:
            final_eps = self.privacy_engine.accountant.get_epsilon(
                delta=self.config.privacy.target_delta
            )
        except Exception:
            final_eps = float('nan')
        
        print("\n" + "=" * 60)
        print("Training Complete")
        if final_eps == final_eps:  # Check for NaN
            print(f"Final (ε={final_eps:.2f}, δ={self.config.privacy.target_delta:.2e})")
        else:
            print(f"Final (ε=N/A, δ={self.config.privacy.target_delta:.2e})")
        print("=" * 60)
        
        return self.metrics_collector.finalize()
    
    def save_results(self, filepath: str):
        """Save results to file"""
        self.metrics_collector.save(filepath)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Use absolute imports when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from inseparable_six.config import quick_test_config
    from inseparable_six.allocation_strategies import (
        UniformStrategy,
        GradientNormStrategy,
        DepthBasedStrategy,
    )
    from inseparable_six.baseline import (
        HuggingFaceTextClassificationDataset,
        GPT2ClassificationModel,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    
    print("\n### LAYER-ADAPTIVE DP OPTIMIZER TEST ###\n")
    
    # Quick test config
    config = quick_test_config(
        use_lora=True,
        use_ghost_clipping=False,  # Layer-adaptive uses standard mode
        max_train_samples=100,
        max_test_samples=50,
    )
    
    # Load model and data
    dataset_adapter = HuggingFaceTextClassificationDataset("imdb")
    model_adapter = GPT2ClassificationModel("gpt2")
    
    model, tokenizer = model_adapter.load_model(
        config.model,
        num_labels=dataset_adapter.get_num_labels(),
        device=config.device,
    )
    
    # Apply LoRA
    model = model_adapter.apply_lora(model, config.lora)
    
    # Prepare for Opacus
    model = ModuleValidator.fix(model)
    
    # Get data
    train_loader, test_loader = dataset_adapter.get_dataloaders(
        tokenizer, config.dataset, config.training
    )
    
    # Test with different strategies
    strategies = [
        ("Uniform", UniformStrategy()),
        ("Depth-Increasing", DepthBasedStrategy(pattern="linear_increasing")),
    ]
    
    results = []
    for name, strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print(f"{'=' * 60}")
        
        # Fresh model for each strategy
        model, _ = model_adapter.load_model(
            config.model,
            num_labels=dataset_adapter.get_num_labels(),
            device=config.device,
        )
        model = model_adapter.apply_lora(model, config.lora)
        model = ModuleValidator.fix(model)
        
        trainer = LayerAdaptiveDPTrainer(config, strategy)
        trainer.setup(model, train_loader, test_loader)
        result = trainer.train()
        
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        print(f"\n{name}:")
        print(f"  Final Acc: {result.final_eval_accuracy:.2f}%")
        print(f"  Final ε: {result.final_epsilon:.2f}")
