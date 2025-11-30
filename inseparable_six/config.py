"""
Configuration classes for Layer-Adaptive DP Training

Provides clean, consistent configuration interfaces for:
- Training parameters
- Privacy parameters  
- Model/LoRA parameters
- Dataset parameters
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import torch


class GradSampleMode(Enum):
    """Gradient sampling modes for DP training"""
    HOOKS = "hooks"      # Standard per-sample gradient computation
    GHOST = "ghost"      # Ghost clipping (memory-efficient, two-pass)


class PrivacyDataSource(Enum):
    """Source of data for gradient estimation in allocation strategies"""
    PUBLIC = "public"           # Use separate public dataset (no privacy cost)
    PRIVATE = "private"         # Use private data (consumes privacy budget)


@dataclass
class PrivacyConfig:
    """Privacy-related configuration parameters"""
    target_epsilon: float = 7.5
    target_delta: Optional[float] = None  # If None, set to 1/n
    max_grad_norm: float = 1.0            # Global clipping bound C_global
    noise_multiplier: Optional[float] = None  # Computed from epsilon if None
    grad_sample_mode: GradSampleMode = GradSampleMode.HOOKS
    accountant_type: str = "rdp"
    secure_mode: bool = False
    
    def __post_init__(self):
        if isinstance(self.grad_sample_mode, str):
            self.grad_sample_mode = GradSampleMode(self.grad_sample_mode)


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration"""
    enabled: bool = True
    rank: int = 8                         # LoRA rank (r)
    alpha: int = 16                       # LoRA scaling factor
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_length: int = 256                 # Max sequence length for tokenization
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    seed: int = 42
    
    # Dataset sampling (for quick testing)
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass 
class ModelConfig:
    """Model configuration"""
    name: str = "gpt2"
    num_labels: int = 2
    cache_dir: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    test_split: str = "test"
    
    # For gradient estimation with public data
    public_dataset_name: Optional[str] = None  # If None, use same as main dataset
    public_data_samples: int = 100             # Samples for gradient estimation


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs"""
    name: str = "dp_experiment"
    
    # Sub-configurations
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # System settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "./outputs"
    save_checkpoints: bool = False
    log_interval: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary"""
        return cls(
            name=config_dict.get("name", "dp_experiment"),
            privacy=PrivacyConfig(**config_dict.get("privacy", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            device=config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            output_dir=config_dict.get("output_dir", "./outputs"),
            save_checkpoints=config_dict.get("save_checkpoints", False),
            log_interval=config_dict.get("log_interval", 10),
        )


# Convenience factory functions for common configurations
def quick_test_config(
    use_lora: bool = True,
    use_ghost_clipping: bool = True,
    max_train_samples: int = 100,
    max_test_samples: int = 50,
) -> ExperimentConfig:
    """Create a configuration for quick testing"""
    return ExperimentConfig(
        name="quick_test",
        privacy=PrivacyConfig(
            target_epsilon=7.5,
            grad_sample_mode=GradSampleMode.GHOST if use_ghost_clipping else GradSampleMode.HOOKS,
        ),
        lora=LoRAConfig(enabled=use_lora),
        training=TrainingConfig(
            batch_size=16,
            epochs=1,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        ),
    )


def full_training_config(
    target_epsilon: float = 3.0,
    use_lora: bool = True,
    epochs: int = 5,
) -> ExperimentConfig:
    """Create a configuration for full training runs"""
    return ExperimentConfig(
        name="full_training",
        privacy=PrivacyConfig(
            target_epsilon=target_epsilon,
            grad_sample_mode=GradSampleMode.GHOST,  # Memory efficient for full runs
        ),
        lora=LoRAConfig(enabled=use_lora),
        training=TrainingConfig(
            batch_size=32,
            epochs=epochs,
        ),
        save_checkpoints=True,
    )
