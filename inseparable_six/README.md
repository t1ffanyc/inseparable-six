# Layer-Adaptive DP Fine-Tuning

Framework comparing uniform vs. layer-adaptive differentially private fine-tuning of LLMs.

## Overview

Implements **layer-adaptive DP-SGD**, assigning per-layer clipping bounds based on signal strength rather than using a uniform bound across all layers. Adapting noise to each layer's sensitivity may improve utility compared to uniform DP noise.

- **Global clipping bound (C_global)**: Overall gradient clipping bound
- **Layer clipping bounds (C_l)**: Per-layer bounds where `sqrt(sum(C_l^2)) = C_global`
- **Noise multiplier (σ)**: Controls privacy-utility tradeoff (same for all layers)

## Installation

### Quick Setup (New Environment)

```bash
# Clone the repository
git clone https://github.com/t1ffanyc/inseparable-six.git
cd inseparable-six/inseparable_six

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate 

# Install the package in editable mode
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

### Using the Package

Once installed, you can import directly, or run tests.

```python
from inseparable_six import (
    ExperimentConfig,
    quick_test_config,
    UniformStrategy,
    GradientNormStrategy,
    DepthBasedStrategy,
)
```

## Package Structure

```
inseparable_six/
├── config.py                   # Configuration dataclasses
├── metrics.py                  # Metrics collection and analysis
├── allocation_strategies.py    # Per-layer budget allocation strategies
├── baseline.py                 # BaselineDPTrainer (extensible OOP interface)
├── layer_adaptive_optimizer.py # LayerAdaptiveDPOptimizer
├── run_experiments.py          # Comprehensive experiment runner
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
└── __init__.py
```

## Running Experiments

### Command Line

```bash
# get full message
python -m inseparable_six.run_experiments --help

# Quick test
python -m inseparable_six.run_experiments --quick-test

# Full experiment suite
python -m inseparable_six.run_experiments --full

# Custom config
python -m inseparable_six.run_experiments --config my_config.json

# With overrides
python -m inseparable_six.run_experiments --epochs 5 --epsilon 4.0 --batch-size 32
```

This runs experiments comparing:
1. Non-DP fine-tuning (baseline utility)
2. DP fine-tuning with various allocation strategies (uniform, depth-based, gradient-norm)

### Using the Python API

```python
from inseparable_six import (
    ExperimentConfig,
    quick_test_config,
    BaselineDPTrainer,
    LayerAdaptiveDPTrainer,
    UniformStrategy,
    GradientNormStrategy,
    DepthBasedStrategy,
    MetricsCollector,
)

# Baseline DP training
config = quick_test_config(use_lora=True)
trainer = BaselineDPTrainer(config).setup()
results = trainer.train()

# Layer-adaptive training with gradient-norm strategy
strategy = GradientNormStrategy(num_estimation_samples=100)
la_trainer = LayerAdaptiveDPTrainer(config, strategy).setup()
la_results = la_trainer.train()
```

## Allocation Strategies

### Uniform Strategy
Allocates equal clipping bounds to all layers:
```python
strategy = UniformStrategy()
```

### Gradient-Norm Strategy
Allocates bounds proportional to estimated gradient norms (higher bounds for larger gradients):
```python
strategy = GradientNormStrategy(
    num_estimation_samples=100,
    data_source=PrivacyDataSource.PRIVATE,  # or PUBLIC
)
```

The `data_source` option controls whether gradient estimation uses the private training data (small privacy cost) or separate public data (no privacy cost).

### Depth-Based Strategies
Allocate based on layer position:

```python
# Linear increasing (deeper layers get more budget)
strategy = DepthBasedStrategy(distribution="linear_increasing")

# Linear decreasing (earlier layers get more budget)
strategy = DepthBasedStrategy(distribution="linear_decreasing")

# Gaussian peak (middle layers get more budget)
strategy = DepthBasedStrategy(distribution="gaussian_peak")

# U-shaped (early and late layers get more budget)
strategy = DepthBasedStrategy(distribution="u_shaped")
```

Or use the factory function:
```python
from inseparable_six import get_strategy, list_strategies

print(list_strategies())
# ['uniform', 'gradient_norm', 'depth_linear_increasing', ...]

strategy = get_strategy("depth_gaussian_peak")
```

## Configuration

Use dataclasses for configuration:

```python
from inseparable_six import (
    ExperimentConfig,
    PrivacyConfig,
    LoRAConfig,
    TrainingConfig,
)

config = ExperimentConfig(
    name="my_experiment",
    privacy=PrivacyConfig(
        target_epsilon=8.0,
        max_grad_norm=1.0,
    ),
    lora=LoRAConfig(
        enabled=True,
        rank=8,
        alpha=16,
    ),
    training=TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=5e-4,
    ),
)
```

## Metrics & Analysis

Results are saved as JSON and can be analyzed:

```python
from inseparable_six import (
    MetricsCollector,
    compare_experiments,
    generate_summary_table,
    extract_training_curves,
)

# Load results
results = [MetricsCollector.load_results(path) for path in result_files]

# Compare experiments
comparison = compare_experiments(results)

# Generate summary table
print(generate_summary_table(results))

# Extract training curves for plotting
curves = extract_training_curves(results)
```

## Extensibility

### Custom Dataset Adapter
```python
from inseparable_six import DatasetAdapter

class MyDatasetAdapter(DatasetAdapter):
    def get_dataloaders(self, tokenizer, config, training_config):
        # Load and prepare your dataset
        return train_loader, test_loader
    
    def get_num_labels(self) -> int:
        return 2
    
    @property
    def name(self) -> str:
        return "my_dataset"
```

### Custom Allocation Strategy
```python
from inseparable_six import AllocationStrategy

class MyStrategy(AllocationStrategy):
    @property
    def name(self) -> str:
        return "my_custom_strategy"
    
    def compute_allocation(self, model, global_bound, **kwargs):
        # Return dict: {param_name: clipping_bound}
        pass
```

## Output

Experiment results include:
- **Training metrics**: loss, accuracy per epoch/step
- **Privacy accounting**: epsilon spent over training
- **Timing**: training time, allocation time
- **Memory**: peak GPU memory usage
- **Allocation details**: per-layer clipping bounds

Results are saved to `./experiment_results/` by default.

## Research Goal

Compare whether **layer-adaptive DP** (where noise is allocated based on layer sensitivity) improves utility over **uniform DP** (same noise across all layers), while maintaining the same privacy guarantees.
