"""
Baseline DP Fine-tuning Trainer

Standard Opacus DP-SGD training with:
- Optional LoRA via PEFT
- Ghost clipping support
- Extensible model and dataset interfaces
- Comprehensive metrics collection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
import numpy as np
import time

from .config import (
    ExperimentConfig, 
    PrivacyConfig, 
    LoRAConfig, 
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    GradSampleMode,
)
from .metrics import MetricsCollector, EpochMetrics, ExperimentResult


# =============================================================================
# Abstract Interfaces for Extensibility
# =============================================================================

class DatasetAdapter(ABC):
    """
    Abstract interface for dataset preparation.
    
    Implement this to support new datasets beyond IMDB.
    """
    
    @abstractmethod
    def get_dataloaders(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DatasetConfig,
        training_config: TrainingConfig,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and test DataLoaders.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        pass
    
    @abstractmethod
    def get_num_labels(self) -> int:
        """Return the number of classification labels"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name for logging"""
        pass


class ModelAdapter(ABC):
    """
    Abstract interface for model preparation.
    
    Implement this to support different model architectures.
    """
    
    @abstractmethod
    def load_model(
        self,
        config: ModelConfig,
        num_labels: int,
        device: str,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the pre-trained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAConfig,
    ) -> PreTrainedModel:
        """Apply LoRA adapters to the model"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging"""
        pass


# =============================================================================
# Default Implementations
# =============================================================================

class HuggingFaceTextClassificationDataset(DatasetAdapter):
    """
    Dataset adapter for HuggingFace text classification datasets.
    
    Supports: imdb, sst2, ag_news, yelp_polarity, etc.
    """
    
    DATASET_INFO = {
        "imdb": {"num_labels": 2, "text_col": "text", "label_col": "label"},
        "sst2": {"num_labels": 2, "text_col": "sentence", "label_col": "label"},
        "ag_news": {"num_labels": 4, "text_col": "text", "label_col": "label"},
        "yelp_polarity": {"num_labels": 2, "text_col": "text", "label_col": "label"},
    }
    
    def __init__(self, dataset_name: str = "imdb"):
        self.dataset_name = dataset_name
        self._info = self.DATASET_INFO.get(dataset_name, {
            "num_labels": 2, "text_col": "text", "label_col": "label"
        })
    
    @property
    def name(self) -> str:
        return self.dataset_name
    
    def get_num_labels(self) -> int:
        return self._info["num_labels"]
    
    def get_dataloaders(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DatasetConfig,
        training_config: TrainingConfig,
    ) -> Tuple[DataLoader, DataLoader]:
        
        # Load dataset
        dataset = load_dataset(config.name)
        
        train_dataset = dataset[config.train_split]
        test_dataset = dataset[config.test_split]
        
        # Optionally limit size
        if training_config.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(min(training_config.max_train_samples, len(train_dataset)))
            )
        if training_config.max_test_samples is not None:
            test_dataset = test_dataset.select(
                range(min(training_config.max_test_samples, len(test_dataset)))
            )
        
        text_col = config.text_column or self._info["text_col"]
        label_col = config.label_column or self._info["label_col"]
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=training_config.max_length,
            )
        
        # Tokenize
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_col] if text_col in train_dataset.column_names else [],
        )
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_col] if text_col in test_dataset.column_names else [],
        )
        
        # Rename label column if needed
        if label_col != "label" and label_col in tokenized_train.column_names:
            tokenized_train = tokenized_train.rename_column(label_col, "label")
            tokenized_test = tokenized_test.rename_column(label_col, "label")
        
        # Set format
        columns = ["input_ids", "attention_mask", "label"]
        tokenized_train.set_format(type="torch", columns=columns)
        tokenized_test.set_format(type="torch", columns=columns)
        
        # Create dataloaders
        train_loader = DataLoader(
            tokenized_train,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        
        test_loader = DataLoader(
            tokenized_test,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        
        print(f"  Train samples: {len(tokenized_train)}")
        print(f"  Test samples: {len(tokenized_test)}")
        
        return train_loader, test_loader


class GPT2ClassificationModel(ModelAdapter):
    """
    Model adapter for GPT-2 based sequence classification.
    """
    
    # LoRA target modules for different model architectures
    LORA_TARGETS = {
        "gpt2": ["c_attn", "c_proj"],
        "bert": ["query", "key", "value", "dense"],
        "roberta": ["query", "key", "value", "dense"],
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    }
    
    def __init__(self, model_type: str = "gpt2"):
        self.model_type = model_type
    
    @property
    def name(self) -> str:
        return self.model_type
    
    def load_model(
        self,
        config: ModelConfig,
        num_labels: int,
        device: str,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            cache_dir=config.cache_dir,
        )
        
        # Handle padding token for GPT-2 style models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.name,
            num_labels=num_labels,
            cache_dir=config.cache_dir,
        )
        
        # Set pad token id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        return model, tokenizer
    
    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAConfig,
    ) -> PreTrainedModel:
        
        # Determine target modules
        target_modules = lora_config.target_modules
        if not target_modules:
            target_modules = self.LORA_TARGETS.get(
                self.model_type, 
                ["c_attn", "c_proj"]  # Default to GPT-2 style
            )
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=target_modules,
            bias=lora_config.bias,
        )
        
        model = get_peft_model(model, peft_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable %: {100 * trainable / total:.2f}%")
        
        return model


# =============================================================================
# Main Trainer Class
# =============================================================================

class BaselineDPTrainer:
    """
    Baseline trainer using standard DP-SGD with Opacus.
    
    Features:
    - Standard DP-SGD or Ghost Clipping
    - Optional LoRA fine-tuning
    - Comprehensive metrics collection
    - Extensible via adapters for models and datasets
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_adapter: Optional[DatasetAdapter] = None,
        model_adapter: Optional[ModelAdapter] = None,
    ):
        self.config = config
        self._set_seed()
        
        # Use provided adapters or create defaults
        self.dataset_adapter = dataset_adapter or HuggingFaceTextClassificationDataset(
            config.dataset.name
        )
        self.model_adapter = model_adapter or GPT2ClassificationModel(
            config.model.name.split("/")[-1].split("-")[0]  # Extract model type
        )
        
        # Components (initialized in setup())
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.criterion: Optional[Callable] = None  # For ghost clipping
        
        # State
        self.is_setup: bool = False
        self.use_ghost_clipping: bool = False
        
        # Metrics
        self.metrics_collector = MetricsCollector(config.name)
        
    def _set_seed(self):
        """Set random seeds for reproducibility"""
        seed = self.config.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup(self) -> "BaselineDPTrainer":
        """
        Initialize all components for training.
        
        Returns self for method chaining.
        """
        print("=" * 60)
        print(f"Setting up: {self.config.name}")
        print("=" * 60)
        
        # Load model and tokenizer
        print(f"\n[1/5] Loading {self.config.model.name} model...")
        num_labels = self.dataset_adapter.get_num_labels()
        self.model, self.tokenizer = self.model_adapter.load_model(
            self.config.model,
            num_labels,
            self.config.device,
        )
        
        # Load dataset
        print(f"\n[2/5] Loading {self.config.dataset.name} dataset...")
        self.train_loader, self.test_loader = self.dataset_adapter.get_dataloaders(
            self.tokenizer,
            self.config.dataset,
            self.config.training,
        )
        
        # Set delta if not specified
        if self.config.privacy.target_delta is None:
            train_size = len(self.train_loader.dataset)
            self.config.privacy.target_delta = 1.0 / train_size
            print(f"  Delta set to 1/n = {self.config.privacy.target_delta:.2e}")
        
        # Apply LoRA if enabled
        if self.config.lora.enabled:
            print(f"\n[3/5] Applying LoRA (rank={self.config.lora.rank})...")
            self.model = self.model_adapter.apply_lora(self.model, self.config.lora)
        else:
            print(f"\n[3/5] Skipping LoRA (full fine-tuning)...")
        
        # Prepare for DP
        print(f"\n[4/5] Preparing model for Opacus...")
        self._prepare_for_dp()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Attach Privacy Engine
        print(f"\n[5/5] Attaching PrivacyEngine...")
        self._attach_privacy_engine()
        
        # Set up metrics collector
        self.metrics_collector.set_config(self.config.to_dict())
        self.metrics_collector.set_model_info(
            model_name=self.config.model.name,
            total_params=sum(p.numel() for p in self.model.parameters()),
            trainable_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            lora_enabled=self.config.lora.enabled,
        )
        
        self.is_setup = True
        
        print("\n" + "=" * 60)
        print("Setup complete! Ready to train.")
        print("=" * 60 + "\n")
        
        return self
    
    def _prepare_for_dp(self):
        """Make model compatible with Opacus"""
        errors = ModuleValidator.validate(self.model, strict=False)
        
        if errors:
            print(f"  Found {len(errors)} incompatible modules, fixing...")
            self.model = ModuleValidator.fix(self.model)
        else:
            print(f"  Model is Opacus-compatible ✓")
    
    def _attach_privacy_engine(self):
        """Attach Opacus PrivacyEngine with specified configuration"""
        self.privacy_engine = PrivacyEngine()
        
        privacy_config = self.config.privacy
        grad_sample_mode = privacy_config.grad_sample_mode.value
        
        result = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config.training.epochs,
            target_epsilon=privacy_config.target_epsilon,
            target_delta=privacy_config.target_delta,
            max_grad_norm=privacy_config.max_grad_norm,
            grad_sample_mode=grad_sample_mode,
        )
        
        # Handle different return signatures
        if grad_sample_mode == "ghost":
            self.model, self.optimizer, self.criterion, self.train_loader = result
            self.use_ghost_clipping = True
        else:
            self.model, self.optimizer, self.train_loader = result
            self.criterion = None
            self.use_ghost_clipping = False
        
        # Store computed noise multiplier
        self.config.privacy.noise_multiplier = self.optimizer.noise_multiplier
        
        print(f"  Target ε: {privacy_config.target_epsilon}")
        print(f"  Target δ: {privacy_config.target_delta:.2e}")
        print(f"  Noise multiplier (σ): {self.optimizer.noise_multiplier:.4f}")
        print(f"  Max grad norm (C): {privacy_config.max_grad_norm}")
        print(f"  Grad sample mode: {grad_sample_mode}")
        print(f"  Ghost clipping: {'ENABLED ✓' if self.use_ghost_clipping else 'DISABLED'}")
    
    def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        Train for one epoch.
        
        Returns EpochMetrics with training statistics.
        """
        if not self.is_setup:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        self.model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.metrics_collector.start_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
        )
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch.get("labels", batch.get("label")).to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (different for ghost clipping)
            if self.use_ghost_clipping:
                # Ghost clipping: use criterion wrapper
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                loss_value = loss.item()
            else:
                # Standard mode
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                loss_value = loss.item()
            
            self.optimizer.step()
            
            # Compute metrics
            total_loss += loss_value
            predictions = torch.argmax(outputs.logits, dim=-1)
            batch_correct = (predictions == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            step_time_ms = (time.time() - step_start) * 1000
            
            # Log step metrics
            self.metrics_collector.log_step(
                step=step,
                epoch=epoch,
                loss=loss_value,
                accuracy=100 * batch_correct / labels.size(0),
                batch_size=labels.size(0),
                learning_rate=self.config.training.learning_rate,
                step_time_ms=step_time_ms,
            )
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'acc': f"{100 * correct / total:.2f}%",
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        epsilon = self.privacy_engine.get_epsilon(self.config.privacy.target_delta)
        epoch_time = time.time() - epoch_start
        
        return EpochMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            train_accuracy=accuracy,
            epsilon=epsilon,
            delta=self.config.privacy.target_delta,
            epoch_time_seconds=epoch_time,
        )
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate on test set.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        if not self.is_setup:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
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
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> ExperimentResult:
        """
        Full training loop.
        
        Returns:
            ExperimentResult with all training metrics
        """
        if not self.is_setup:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        
        print("\nStarting training...\n")
        self.metrics_collector.start_experiment()
        
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_loss, eval_acc = self.evaluate()
            
            # Finalize epoch metrics
            self.metrics_collector.end_epoch(
                train_loss=train_metrics.train_loss,
                train_accuracy=train_metrics.train_accuracy,
                eval_accuracy=eval_acc,
                eval_loss=eval_loss,
                epsilon=train_metrics.epsilon,
                delta=train_metrics.delta,
            )
            
            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics.train_loss:.4f}")
            print(f"  Train Acc: {train_metrics.train_accuracy:.2f}%")
            print(f"  Eval Acc: {eval_acc:.2f}%")
            print(f"  Privacy (ε): {train_metrics.epsilon:.2f}")
            print()
        
        # Finalize
        final_epsilon = self.privacy_engine.get_epsilon(self.config.privacy.target_delta)
        
        print("=" * 60)
        print("Training Complete")
        print(f"Final Privacy: (ε={final_epsilon:.2f}, δ={self.config.privacy.target_delta:.2e})")
        print("=" * 60)
        
        return self.metrics_collector.finalize()
    
    def save_results(self, filepath: str):
        """Save training results to JSON file"""
        self.metrics_collector.save(filepath)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trainer(
    use_lora: bool = True,
    use_ghost_clipping: bool = True,
    target_epsilon: float = 7.5,
    batch_size: int = 16,
    epochs: int = 3,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    model_name: str = "gpt2",
    dataset_name: str = "imdb",
) -> BaselineDPTrainer:
    """
    Convenience function to create a trainer with common settings.
    """
    config = ExperimentConfig(
        name=f"baseline_{'lora' if use_lora else 'full'}_{'ghost' if use_ghost_clipping else 'hooks'}",
        privacy=PrivacyConfig(
            target_epsilon=target_epsilon,
            grad_sample_mode=GradSampleMode.GHOST if use_ghost_clipping else GradSampleMode.HOOKS,
        ),
        lora=LoRAConfig(enabled=use_lora),
        training=TrainingConfig(
            batch_size=batch_size,
            epochs=epochs,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        ),
        model=ModelConfig(name=model_name),
        dataset=DatasetConfig(name=dataset_name),
    )
    
    return BaselineDPTrainer(config)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Use absolute imports when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from inseparable_six.config import quick_test_config
    
    print("\n### BASELINE DP TRAINING ###\n")
    
    # Quick test with LoRA and ghost clipping
    config = quick_test_config(
        use_lora=True,
        use_ghost_clipping=True,
        max_train_samples=100,
        max_test_samples=50,
    )
    
    trainer = BaselineDPTrainer(config)
    trainer.setup()
    results = trainer.train()
    
    # Save results
    trainer.save_results("baseline_results.json")
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {results.final_train_accuracy:.2f}%")
    print(f"  Eval Accuracy: {results.final_eval_accuracy:.2f}%")
    print(f"  Final Epsilon: {results.final_epsilon:.2f}")
