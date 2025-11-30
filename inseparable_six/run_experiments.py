#!/usr/bin/env python3
"""
Experiment Runner for Layer-Adaptive DP Fine-Tuning

This script runs comprehensive experiments comparing:
1. Non-DP fine-tuning (baseline utility)
2. DP fine-tuning with various allocation strategies (including uniform)

Tracks: accuracy, loss, privacy spent, training time, memory usage

Usage:
    python run_experiments.py --quick-test           # Quick test run
    python run_experiments.py --full                  # Full experiment suite
    python run_experiments.py --config config.json    # Custom config
"""

import argparse
import json
import os
import sys
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Suppress specific warnings before importing transformers/peft/opacus
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*fan_in_fan_out is set to False.*")
warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from inseparable_six.metrics import (
    MetricsCollector,
    ExperimentResult,
    AllocationMetrics,
    compare_experiments,
    save_comparison,
    generate_summary_table,
    generate_all_plots,
    get_peak_gpu_memory_mb,
    reset_peak_memory_stats,
)
from inseparable_six.config import PrivacyDataSource
from inseparable_six.allocation_strategies import (
    AllocationStrategy,
    UniformStrategy,
    GradientNormStrategy,
    DepthBasedStrategy,
    get_strategy,
    list_strategies,
)
from inseparable_six.layer_adaptive_optimizer import (
    make_layer_adaptive_private,
    LayerAdaptiveDPOptimizer,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentSuiteConfig:
    """Configuration for a suite of experiments"""
    
    # Experiment name
    name: str = "dp_finetuning_comparison"
    output_dir: str = "./experiment_results"
    
    # Model settings
    model_name: str = "gpt2"
    
    # Dataset settings
    dataset_name: str = "imdb"
    max_train_samples: Optional[int] = None  # None = use all
    max_test_samples: Optional[int] = None
    max_length: int = 256
    
    # Training settings
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    seed: int = 42
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Privacy settings
    target_epsilon: float = 4.0
    target_delta: Optional[float] = None  # 1/n if None
    max_grad_norm: float = 1.0
    
    # Gradient estimation for adaptive strategies
    gradient_estimation_samples: int = 100
    use_public_data_for_estimation: bool = False
    
    # Which experiments to run
    run_no_dp: bool = True
    run_strategies: List[str] = field(default_factory=lambda: [
        "uniform",
        "depth_linear_increasing",
        "depth_linear_decreasing",
        "gradient_norm",
    ])
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSuiteConfig":
        return cls(**d)
    
    @classmethod
    def quick_test(cls) -> "ExperimentSuiteConfig":
        """Quick test configuration for debugging"""
        return cls(
            name="quick_test",
            max_train_samples=50,
            max_test_samples=25,
            epochs=1,
            batch_size=8,
            gradient_estimation_samples=25,
            run_strategies=["uniform", "depth_linear_increasing"],
        )
    
    @classmethod
    def full_suite(cls) -> "ExperimentSuiteConfig":
        """Full experiment suite"""
        return cls(
            name="full_comparison",
            epochs=3,
            batch_size=16,
            run_strategies=[
                "uniform",
                "depth_linear_increasing",
                "depth_linear_decreasing", 
                "depth_gaussian_peak",
                "depth_u_shaped",
                "gradient_norm",
            ],
        )


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    config: ExperimentSuiteConfig,
    tokenizer,
) -> Tuple[DataLoader, DataLoader, int]:
    """Load and prepare dataset"""
    print(f"\n[Loading {config.dataset_name} dataset...]")
    
    dataset = load_dataset(config.dataset_name)
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Limit samples if specified
    if config.max_train_samples:
        train_data = train_data.select(range(min(config.max_train_samples, len(train_data))))
    if config.max_test_samples:
        test_data = test_data.select(range(min(config.max_test_samples, len(test_data))))
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )
    
    train_tokenized = train_data.map(tokenize, batched=True, remove_columns=["text"])
    test_tokenized = test_data.map(tokenize, batched=True, remove_columns=["text"])
    
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(
        train_tokenized, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=default_data_collator,
    )
    test_loader = DataLoader(
        test_tokenized,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    
    print(f"  Train samples: {len(train_tokenized)}")
    print(f"  Test samples: {len(test_tokenized)}")
    
    return train_loader, test_loader, len(train_tokenized)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(config: ExperimentSuiteConfig, num_labels: int = 2):
    """Load model and tokenizer"""
    print(f"\n[Loading {config.model_name} model...]")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    return model, tokenizer


def apply_lora(model, config: ExperimentSuiteConfig):
    """Apply LoRA to model"""
    print("\n[Applying LoRA...]")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


def make_opacus_compatible(model):
    """Make model compatible with Opacus"""
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"  Fixing {len(errors)} incompatible modules...")
        model = ModuleValidator.fix(model)
    return model


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
    metrics_collector: MetricsCollector,
    privacy_engine=None,
    target_delta: float = 1e-5,
):
    """Train for one epoch"""
    model.train()
    metrics_collector.start_epoch(epoch)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for step, batch in enumerate(pbar):
        step_start = time.time()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", batch.get("label")).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Metrics
        loss_val = loss.item()
        total_loss += loss_val
        preds = torch.argmax(outputs.logits, dim=-1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        
        step_time = (time.time() - step_start) * 1000
        
        # Get current epsilon if DP
        epsilon = None
        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(target_delta)
        
        metrics_collector.log_step(
            step=step,
            epoch=epoch,
            loss=loss_val,
            accuracy=100 * batch_correct / labels.size(0),
            batch_size=labels.size(0),
            learning_rate=optimizer.param_groups[0]['lr'],
            epsilon=epsilon,
            step_time_ms=step_time,
        )
        
        pbar.set_postfix({'loss': f'{loss_val:.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, device: str):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", batch.get("label")).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(test_loader), 100 * correct / total


# =============================================================================
# Experiment Runners
# =============================================================================

def run_no_dp_experiment(
    config: ExperimentSuiteConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> ExperimentResult:
    """Run non-DP fine-tuning experiment"""
    exp_name = "no_dp_lora"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    reset_peak_memory_stats()
    
    # Load fresh model
    model, _ = load_model(config)
    model = apply_lora(model, config)
    
    # Metrics
    collector = MetricsCollector(exp_name)
    collector.set_config(config.to_dict())
    collector.set_experiment_type("no_dp", "none")
    collector.set_model_info(
        model_name=config.model_name,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        lora_enabled=True,
    )
    collector.start_experiment()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Training
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, config.device,
            epoch, config.epochs, collector,
        )
        eval_loss, eval_acc = evaluate(model, test_loader, config.device)
        
        collector.end_epoch(
            train_loss=train_loss,
            train_accuracy=train_acc,
            eval_loss=eval_loss,
            eval_accuracy=eval_acc,
            epsilon=0.0,  # No DP
            delta=0.0,
        )
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Eval Acc={eval_acc:.2f}%")
    
    return collector.finalize()


def run_dp_experiment(
    config: ExperimentSuiteConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_size: int,
    strategy_name: str,
) -> ExperimentResult:
    """Run DP fine-tuning experiment with specified allocation strategy"""
    exp_name = f"dp_{strategy_name}_lora"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    reset_peak_memory_stats()
    
    # Load fresh model
    model, _ = load_model(config)
    model = apply_lora(model, config)
    model = make_opacus_compatible(model)
    
    # Delta
    target_delta = config.target_delta or (1.0 / train_size)
    
    # Create strategy
    if strategy_name == "gradient_norm":
        strategy = GradientNormStrategy(
            num_estimation_samples=config.gradient_estimation_samples,
            data_source=(
                PrivacyDataSource.PUBLIC if config.use_public_data_for_estimation 
                else PrivacyDataSource.PRIVATE
            ),
            verbose=True,
        )
    else:
        strategy = get_strategy(strategy_name)
    
    print(f"  Strategy: {strategy.name}")
    
    # Metrics
    collector = MetricsCollector(exp_name)
    collector.set_config(config.to_dict())
    collector.set_experiment_type("dp", strategy.name)
    collector.set_model_info(
        model_name=config.model_name,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        lora_enabled=True,
    )
    collector.start_experiment()
    
    # Optimizer
    base_optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Setup layer-adaptive DP (works for uniform strategy too)
    allocation_start = time.time()
    model, optimizer, dp_train_loader, privacy_engine, allocation_metrics = make_layer_adaptive_private(
        model=model,
        optimizer=base_optimizer,
        data_loader=train_loader,
        allocation_strategy=strategy,
        target_epsilon=config.target_epsilon,
        target_delta=target_delta,
        epochs=config.epochs,
        global_clipping_bound=config.max_grad_norm,
        device=config.device,
        verbose=True,
    )
    allocation_metrics.allocation_time_seconds = time.time() - allocation_start
    
    collector.set_allocation_metrics(allocation_metrics)
    
    print(f"  Noise multiplier: {optimizer.noise_multiplier:.4f}")
    print(f"  Target (ε={config.target_epsilon}, δ={target_delta:.2e})")
    
    # Training
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(
            model, dp_train_loader, optimizer, config.device,
            epoch, config.epochs, collector,
            privacy_engine=privacy_engine,
            target_delta=target_delta,
        )
        eval_loss, eval_acc = evaluate(model, test_loader, config.device)
        
        epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)
        
        collector.end_epoch(
            train_loss=train_loss,
            train_accuracy=train_acc,
            eval_loss=eval_loss,
            eval_accuracy=eval_acc,
            epsilon=epsilon,
            delta=target_delta,
        )
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Eval Acc={eval_acc:.2f}%, ε={epsilon:.2f}")
    
    return collector.finalize()


# =============================================================================
# Main Experiment Suite
# =============================================================================

def run_experiment_suite(config: ExperimentSuiteConfig) -> List[ExperimentResult]:
    """Run full experiment suite"""
    print("\n" + "="*70)
    print("LAYER-ADAPTIVE DP FINE-TUNING EXPERIMENT SUITE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Target ε: {config.target_epsilon}")
    print(f"  Device: {config.device}")
    
    results: List[ExperimentResult] = []
    
    # Load tokenizer once for data loading
    _, tokenizer = load_model(config)
    
    # Load data
    train_loader, test_loader, train_size = load_data(config, tokenizer)
    
    # 1. Non-DP baseline (with LoRA for fair comparison)
    if config.run_no_dp:
        try:
            result = run_no_dp_experiment(config, train_loader, test_loader)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in no_dp experiment: {e}")
    
    # 2. DP strategies (includes uniform and all adaptive strategies)
    for strategy_name in config.run_strategies:
        try:
            result = run_dp_experiment(
                config, train_loader, test_loader, train_size, strategy_name
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR in {strategy_name} experiment: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def save_results(
    results: List[ExperimentResult], 
    config: ExperimentSuiteConfig,
):
    """Save all experiment results and generate plots"""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual results
    for result in results:
        filepath = output_dir / f"{result.experiment_name}_{timestamp}.json"
        MetricsCollector._save_results(result, str(filepath))
        print(f"Saved: {filepath}")
    
    # Save comparison
    comparison = compare_experiments(results)
    comparison_path = output_dir / f"comparison_{timestamp}.json"
    save_comparison(comparison, str(comparison_path))
    print(f"Saved: {comparison_path}")
    
    # Save config
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved: {config_path}")
    
    # Generate and save plots
    print("\n[Generating plots...]")
    plot_files = generate_all_plots(results, str(output_dir), timestamp)
    for plot_file in plot_files:
        print(f"Saved: {plot_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(generate_summary_table(results))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run DP fine-tuning experiments")
    
    parser.add_argument("--quick-test", action="store_true", help="Quick test run")
    parser.add_argument("--full", action="store_true", help="Full experiment suite")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--output-dir", type=str, default="./experiment_results")
    
    # Override options
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epsilon", type=float, help="Target epsilon")
    parser.add_argument("--max-train-samples", type=int, help="Max training samples")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            config = ExperimentSuiteConfig.from_dict(json.load(f))
    elif args.quick_test:
        config = ExperimentSuiteConfig.quick_test()
    elif args.full:
        config = ExperimentSuiteConfig.full_suite()
    else:
        config = ExperimentSuiteConfig()
    
    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epsilon:
        config.target_epsilon = args.epsilon
    if args.max_train_samples:
        config.max_train_samples = args.max_train_samples
    if args.model:
        config.model_name = args.model
    if args.dataset:
        config.dataset_name = args.dataset
    
    # Run experiments
    results = run_experiment_suite(config)
    
    # Save results
    if results:
        save_results(results, config)
    else:
        print("\nNo experiments completed successfully.")
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
