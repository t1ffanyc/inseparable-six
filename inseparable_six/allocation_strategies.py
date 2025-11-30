
"""
Signal Strength Strategies for Layer-Adaptive DP

Computes allocation ratios {alpha_l} for per-layer clipping bounds C_l = C_global * alpha_l
Maintains constraint: sqrt(sum(C_l^2)) = C_global
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class AllocationStrategy(ABC):
    """
    Abstract base class for computing per-layer noise allocation.
    
    Returns allocation ratios {alpha_l} where:
    - C_l = C_global * alpha_l (per-layer clipping bound)
    - Constraint: sqrt(sum(alpha_l^2)) = 1.0 (so sqrt(sum(C_l^2)) = C_global)
    """
    
    @abstractmethod
    def compute_allocation(
        self,
        model: nn.Module,
        C_global: float,
        sample_data: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Compute per-layer clipping bounds C_l.
        
        Args:
            model: The model (potentially with LoRA)
            C_global: Global clipping bound
            sample_data: Optional dataloader for signal strength estimation
            device: Device to run computations on
            
        Returns:
            Dictionary mapping param_group_name -> C_l
            where sqrt(sum(C_l^2)) = C_global
        """
        pass
    
    def _get_param_groups(self, model: nn.Module) -> Dict[str, List[nn.Parameter]]:
        """
        Group parameters by transformer block.
        
        For LoRA models: Groups all LoRA parameters per block.
        For full models: Groups all parameters per block.
        
        NOTE: Currently groups by transformer block. Future work could explore:
        - Per-module grouping (attention vs FFN)
        - Per-matrix grouping (LoRA_A vs LoRA_B)
        """
        param_groups = OrderedDict()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Extract block identifier
            # Examples:
            #   "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight" -> "block_0"
            #   "transformer.h.3.mlp.c_fc.weight" -> "block_3"
            
            if ".h." in name:  # GPT-2 style transformer blocks
                parts = name.split(".h.")
                if len(parts) > 1:
                    block_num = parts[1].split(".")[0]
                    group_name = f"block_{block_num}"
                else:
                    group_name = "other"
            else:
                # Non-transformer parameters (embeddings, classification head, etc.)
                group_name = "other"
            
            if group_name not in param_groups:
                param_groups[group_name] = []
            param_groups[group_name].append(param)
        
        return param_groups
    
    def _normalize_allocation(self, raw_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize allocation ratios to satisfy constraint: sqrt(sum(alpha_l^2)) = 1.0
        
        Args:
            raw_allocations: Unnormalized allocation values
            
        Returns:
            Normalized allocations where sqrt(sum(alpha_l^2)) = 1.0
        """
        # Compute normalization factor
        sum_squares = sum(v**2 for v in raw_allocations.values())
        norm_factor = np.sqrt(sum_squares)
        
        if norm_factor == 0:
            # Fallback to uniform if all zeros
            n = len(raw_allocations)
            return {k: 1.0 / np.sqrt(n) for k in raw_allocations.keys()}
        
        # Normalize: divide each by norm_factor so sqrt(sum((v/norm)**2)) = 1
        normalized = {k: v / norm_factor for k, v in raw_allocations.items()}
        
        return normalized
    
    def _allocations_to_clipping_bounds(
        self, 
        normalized_allocations: Dict[str, float],
        C_global: float
    ) -> Dict[str, float]:
        """
        Convert normalized allocation ratios to actual clipping bounds.
        
        C_l = C_global * alpha_l
        """
        return {k: C_global * alpha for k, alpha in normalized_allocations.items()}


class UniformStrategy(AllocationStrategy):
    """
    Baseline: Uniform allocation across all parameter groups.
    Equivalent to standard DP-SGD.
    """
    
    def compute_allocation(
        self,
        model: nn.Module,
        C_global: float,
        sample_data: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        param_groups = self._get_param_groups(model)
        n_groups = len(param_groups)
        
        # Uniform: each group gets equal allocation ratio
        # alpha_l = 1/sqrt(n) so that sqrt(sum(alpha_l^2)) = sqrt(n * 1/n) = 1
        alpha = 1.0 / np.sqrt(n_groups)
        raw_allocations = {name: alpha for name in param_groups.keys()}
        
        # Already normalized, but use helper for consistency
        normalized = self._normalize_allocation(raw_allocations)
        clipping_bounds = self._allocations_to_clipping_bounds(normalized, C_global)
        
        return clipping_bounds


class GradientNormStrategy(AllocationStrategy):
    """
    Signal-proportional allocation based on observed gradient magnitudes.
    
    Layers with larger gradients receive larger clipping bounds (more "budget").
    Hypothesis: Layers with stronger signals benefit more from looser clipping.
    """
    
    def __init__(self, n_samples: int = 100):
        """
        Args:
            n_samples: Number of samples to use for gradient estimation
        """
        self.n_samples = n_samples
    
    def compute_allocation(
        self,
        model: nn.Module,
        C_global: float,
        sample_data: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        if sample_data is None:
            raise ValueError("GradientNormStrategy requires sample_data for estimation")
        
        param_groups = self._get_param_groups(model)
        
        # Accumulate gradient norms per group
        group_grad_norms = {name: 0.0 for name in param_groups.keys()}
        
        model.train()
        model.to(device)
        
        # Compute number of samples to use
        dataset_size = len(sample_data.dataset)
        n_samples = min(self.n_samples, dataset_size)
        
        print(f"  Computing gradient norms on {n_samples} samples...")
        
        samples_processed = 0
        with tqdm(total=n_samples, desc="  Gradient estimation") as pbar:
            for batch in sample_data:
                if samples_processed >= n_samples:
                    break
                    
                # Forward pass
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # default_data_collator uses "labels" key
                labels = batch.get("labels", batch.get("label")).to(device)
                
                # Limit batch if we only need partial
                batch_size = input_ids.size(0)
                remaining = n_samples - samples_processed
                if batch_size > remaining:
                    input_ids = input_ids[:remaining]
                    attention_mask = attention_mask[:remaining]
                    labels = labels[:remaining]
                    batch_size = remaining
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradient norms per group
                for group_name, params in param_groups.items():
                    group_norm = 0.0
                    for param in params:
                        if param.grad is not None:
                            group_norm += param.grad.norm(2).item() ** 2
                    group_grad_norms[group_name] += np.sqrt(group_norm)
                
                # Clear gradients
                model.zero_grad()
                
                samples_processed += batch_size
                pbar.update(batch_size)
        
        # Average across samples
        for group_name in group_grad_norms:
            group_grad_norms[group_name] /= n_samples
        
        # Normalize to get allocation ratios
        normalized = self._normalize_allocation(group_grad_norms)
        clipping_bounds = self._allocations_to_clipping_bounds(normalized, C_global)
        
        return clipping_bounds


class DepthBasedStrategy(AllocationStrategy):
    """
    Allocation based on layer depth in the architecture.
    
    Supports different distribution patterns:
    - 'uniform': Equal allocation (same as UniformStrategy)
    - 'linear_increasing': Linearly increases with depth
    - 'linear_decreasing': Linearly decreases with depth
    - 'gaussian_peak': Gaussian centered at middle layers
    - 'u_shaped': Higher at edges, lower in middle
    """
    
    def __init__(self, pattern: str = "linear_increasing"):
        """
        Args:
            pattern: Distribution pattern for allocation
        """
        valid_patterns = [
            'uniform', 'linear_increasing', 'linear_decreasing',
            'gaussian_peak', 'u_shaped'
        ]
        if pattern not in valid_patterns:
            raise ValueError(f"Pattern must be one of {valid_patterns}")
        
        self.pattern = pattern
    
    def compute_allocation(
        self,
        model: nn.Module,
        C_global: float,
        sample_data: Optional[DataLoader] = None,
        device: str = "cuda"
    ) -> Dict[str, float]:
        
        param_groups = self._get_param_groups(model)
        
        # Extract block indices (assumes naming like "block_0", "block_1", etc.)
        block_items = []
        other_items = []
        
        for name in param_groups.keys():
            if name.startswith("block_"):
                try:
                    block_idx = int(name.split("_")[1])
                    block_items.append((name, block_idx))
                except:
                    other_items.append(name)
            else:
                other_items.append(name)
        
        # Sort blocks by index
        block_items.sort(key=lambda x: x[1])
        n_blocks = len(block_items)
        
        if n_blocks == 0:
            # No transformer blocks found, fall back to uniform
            print("  Warning: No transformer blocks found, using uniform allocation")
            alpha = 1.0 / np.sqrt(len(param_groups))
            return {name: C_global * alpha for name in param_groups.keys()}
        
        # Compute allocation based on pattern
        raw_allocations = {}
        
        if self.pattern == 'uniform':
            # Equal allocation
            for name, _ in block_items:
                raw_allocations[name] = 1.0
        
        elif self.pattern == 'linear_increasing':
            # Increases linearly: earlier layers get less, later layers get more
            for name, idx in block_items:
                raw_allocations[name] = (idx + 1) / n_blocks
        
        elif self.pattern == 'linear_decreasing':
            # Decreases linearly: earlier layers get more, later layers get less
            for name, idx in block_items:
                raw_allocations[name] = (n_blocks - idx) / n_blocks
        
        elif self.pattern == 'gaussian_peak':
            # Gaussian centered at middle layers
            center = n_blocks / 2
            sigma = n_blocks / 4  # Adjust spread
            for name, idx in block_items:
                raw_allocations[name] = np.exp(-((idx - center) ** 2) / (2 * sigma ** 2))
        
        elif self.pattern == 'u_shaped':
            # Higher at edges (early and late layers), lower in middle
            center = n_blocks / 2
            for name, idx in block_items:
                # Distance from center, normalized
                dist_from_center = abs(idx - center) / (n_blocks / 2)
                raw_allocations[name] = dist_from_center
        
        # Handle non-block parameters (embeddings, classification head, etc.)
        # Give them average allocation
        if other_items:
            avg_block_allocation = np.mean(list(raw_allocations.values()))
            for name in other_items:
                raw_allocations[name] = avg_block_allocation
        
        # Normalize and convert to clipping bounds
        normalized = self._normalize_allocation(raw_allocations)
        clipping_bounds = self._allocations_to_clipping_bounds(normalized, C_global)
        
        return clipping_bounds


class InverseDepthStrategy(DepthBasedStrategy):
    """
    Inverse of depth-based strategy for comparison.
    
    If depth-based uses 'linear_increasing', this uses 'linear_decreasing', etc.
    """
    
    INVERSE_PATTERNS = {
        'linear_increasing': 'linear_decreasing',
        'linear_decreasing': 'linear_increasing',
        'gaussian_peak': 'u_shaped',
        'u_shaped': 'gaussian_peak',
        'uniform': 'uniform'
    }
    
    def __init__(self, base_pattern: str = "linear_increasing"):
        inverse_pattern = self.INVERSE_PATTERNS.get(base_pattern, 'uniform')
        super().__init__(pattern=inverse_pattern)
        self.base_pattern = base_pattern


# Utility function for comparing strategies
def compare_strategies(
    model: nn.Module,
    strategies: List[Tuple[str, AllocationStrategy]],
    C_global: float = 1.0,
    sample_data: Optional[DataLoader] = None,
    device: str = "cuda"
):
    """
    Compare allocation results across different strategies.
    
    Args:
        model: The model to analyze
        strategies: List of (name, strategy) tuples
        C_global: Global clipping bound
        sample_data: Optional sample data for gradient-based strategies
        device: Device for computations
    """
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for strategy_name, strategy in strategies:
        print(f"\n[{strategy_name}]")
        allocations = strategy.compute_allocation(model, C_global, sample_data, device)
        results[strategy_name] = allocations
        
        # Verify constraint
        sum_squares = sum(v**2 for v in allocations.values())
        constraint_check = np.sqrt(sum_squares)
        
        print(f"  Constraint check: sqrt(sum(C_l^2)) = {constraint_check:.6f} (target: {C_global:.6f})")
        print(f"  Number of groups: {len(allocations)}")
        print(f"  C_l range: [{min(allocations.values()):.4f}, {max(allocations.values()):.4f}]")
        
        # Show allocation for first few groups
        print(f"  Sample allocations:")
        for i, (name, c_l) in enumerate(list(allocations.items())[:5]):
            print(f"    {name}: C_l = {c_l:.4f}")
        if len(allocations) > 5:
            print(f"    ... ({len(allocations) - 5} more groups)")
    
    print("\n" + "=" * 70)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Signal Strength Strategies...")
    from baseline import BaselineConfig, BaselineDPTrainer
    
    # Setup model and data
    config = BaselineConfig(use_lora=True, batch_size=16)
    trainer = BaselineDPTrainer(config)
    trainer.setup()
    
    # Compare strategies
    strategies = [
        ("Uniform", UniformStrategy()),
        ("Gradient-Norm", GradientNormStrategy(n_samples=50)),
        ("Depth-Increasing", DepthBasedStrategy(pattern="linear_increasing")),
        ("Depth-Decreasing", DepthBasedStrategy(pattern="linear_decreasing")),
        ("Gaussian-Peak", DepthBasedStrategy(pattern="gaussian_peak")),
    ]
    
    compare_strategies(
        trainer.model,
        strategies,
        C_global=1.0,
        sample_data=trainer.train_loader,
        device=config.device
    )