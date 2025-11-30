"""
Inseparable Six: Layer-Adaptive Differential Privacy for LLM Fine-Tuning

This package implements layer-adaptive DP-SGD for fine-tuning large language models
with differential privacy. Key features:

- Per-layer clipping bound allocation based on signal strength
- Multiple allocation strategies (uniform, gradient-norm, depth-based)
- Support for LoRA fine-tuning with Opacus
- Ghost clipping for memory efficiency
- Comprehensive metrics collection for analysis

Main Components:
- config: Configuration classes for experiments
- metrics: Data collection and serialization
- allocation_strategies: Per-layer budget allocation strategies
- baseline: Standard DP-SGD training
- layer_adaptive_optimizer: Layer-adaptive DP optimizer

Example Usage:
    from inseparable_six import (
        ExperimentConfig,
        quick_test_config,
        BaselineDPTrainer,
        LayerAdaptiveDPTrainer,
        UniformStrategy,
        GradientNormStrategy,
        DepthBasedStrategy,
    )
    
    # Quick test with baseline
    config = quick_test_config(use_lora=True, use_ghost_clipping=True)
    trainer = BaselineDPTrainer(config).setup()
    results = trainer.train()
    
    # Layer-adaptive training
    strategy = GradientNormStrategy(num_estimation_samples=100)
    la_trainer = LayerAdaptiveDPTrainer(config, strategy)
    # ... setup and train
"""

__version__ = "0.1.0"

# Configuration
from .config import (
    ExperimentConfig,
    PrivacyConfig,
    LoRAConfig,
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    GradSampleMode,
    PrivacyDataSource,
    quick_test_config,
    full_training_config,
)

# Metrics
from .metrics import (
    MetricsCollector,
    ExperimentResult,
    EpochMetrics,
    StepMetrics,
    AllocationMetrics,
    extract_training_curves,
    extract_allocation_data,
    compare_experiments,
    save_comparison,
    load_comparison,
    generate_summary_table,
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_peak_memory_stats,
)

# Allocation Strategies
from .allocation_strategies import (
    AllocationStrategy,
    UniformStrategy,
    GradientNormStrategy,
    DepthBasedStrategy,
    InverseDepthStrategy,
    CustomAllocationStrategy,
    get_strategy,
    list_strategies,
    compare_strategies,
)

# Baseline Trainer
from .baseline import (
    BaselineDPTrainer,
    DatasetAdapter,
    ModelAdapter,
    HuggingFaceTextClassificationDataset,
    GPT2ClassificationModel,
    create_trainer,
)

# Layer-Adaptive Components
from .layer_adaptive_optimizer import (
    LayerAdaptiveDPOptimizer,
    LayerClippingConfig,
    LayerAdaptiveDPTrainer,
    make_layer_adaptive_private,
    create_layer_configs,
)

__all__ = [
    # Version
    "__version__",
    
    # Config
    "ExperimentConfig",
    "PrivacyConfig",
    "LoRAConfig",
    "TrainingConfig",
    "ModelConfig",
    "DatasetConfig",
    "GradSampleMode",
    "PrivacyDataSource",
    "quick_test_config",
    "full_training_config",
    
    # Metrics
    "MetricsCollector",
    "ExperimentResult",
    "EpochMetrics",
    "StepMetrics",
    "AllocationMetrics",
    "extract_training_curves",
    "extract_allocation_data",
    "compare_experiments",
    "save_comparison",
    "load_comparison",
    "generate_summary_table",
    "get_gpu_memory_mb",
    "get_peak_gpu_memory_mb",
    "reset_peak_memory_stats",
    
    # Strategies
    "AllocationStrategy",
    "UniformStrategy",
    "GradientNormStrategy",
    "DepthBasedStrategy",
    "InverseDepthStrategy",
    "CustomAllocationStrategy",
    "get_strategy",
    "list_strategies",
    "compare_strategies",
    
    # Baseline
    "BaselineDPTrainer",
    "DatasetAdapter",
    "ModelAdapter",
    "HuggingFaceTextClassificationDataset",
    "GPT2ClassificationModel",
    "create_trainer",
    
    # Layer-Adaptive
    "LayerAdaptiveDPOptimizer",
    "LayerClippingConfig",
    "LayerAdaptiveDPTrainer",
    "make_layer_adaptive_private",
    "create_layer_configs",
]
