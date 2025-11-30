"""
Baseline DP Fine-tuning
- Standard Opacus DP-SGD training
- Optional LoRA via PEFT
- Text classification with GPT-2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2ForSequenceClassification,
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
import numpy as np


class BaselineConfig:
    """Configuration for baseline DP training"""
    def __init__(
        self,
        model_name: str = "gpt2",
        dataset_name: str = "imdb",
        use_lora: bool = True,
        
        # LoRA parameters
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        
        # Training parameters
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 5e-4,
        max_length: int = 256,
        
        # DP parameters
        target_epsilon: float = 7.5,
        target_delta: float = None,  # Will be set to 1/len(dataset)
        max_grad_norm: float = 1.0,
        grad_sample_mode: str = "ghost",  # "hooks" (default) or "ghost" for ghost clipping
        
        # Dataset sampling, can limit for faster testing
        max_train_samples: int = None,  # Limit training samples (None = use all)
        max_test_samples: int = None,   # Limit test samples (None = use all)
        
        # System
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.grad_sample_mode = grad_sample_mode
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.device = device
        self.seed = seed


class BaselineDPTrainer:
    """Baseline trainer using standard DP-SGD"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self._set_seed()
        
        # initialized in setup()
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.privacy_engine = None
        self.criterion = None  # Used for ghost clipping mode
        self.use_ghost_clipping = False
        
    def _set_seed(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def setup(self):
        """Initialize model, data, and DP components"""
        print("=" * 50)
        
        # Load dataset
        print(f"\n[1/5] Loading {self.config.dataset_name} dataset...")
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Set delta based on dataset size
        if self.config.target_delta is None:
            train_size = len(self.train_loader.dataset)
            self.config.target_delta = 1.0 / train_size
            print(f"  → Delta set to 1/n = {self.config.target_delta:.2e}")
        
        # Load model
        print(f"\n[2/5] Loading {self.config.model_name} model...")
        self.model, self.tokenizer = self._prepare_model()
        
        # Apply LoRA if requested
        if self.config.use_lora:
            print(f"\n[3/5] Applying LoRA (r={self.config.lora_r})...")
            self.model = self._apply_lora()
        else:
            print(f"\n[3/5] Skipping LoRA (full fine-tuning)...")
        
        # Prepare for DP
        print(f"\n[4/5] Preparing model for Opacus...")
        self._prepare_for_dp()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Attach Privacy Engine
        print(f"\n[5/5] Attaching PrivacyEngine...")
        self._attach_privacy_engine()
        
        print("\n" + "=" * 50)
        print("Setup complete! Ready to train.")
        print("=" * 50 + "\n")
    
    def _prepare_data(self):
        """Load and tokenize dataset"""
        # Load IMDB dataset
        dataset = load_dataset(self.config.dataset_name)
        
        # Optionally limit dataset size for faster testing
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        
        if self.config.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(self.config.max_train_samples, len(train_dataset))))
        if self.config.max_test_samples is not None:
            test_dataset = test_dataset.select(range(min(self.config.max_test_samples, len(test_dataset))))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length
            )
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize
        tokenized_train = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Format for PyTorch (note: default_data_collator uses "labels" not "label")
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        # Create dataloaders
        train_loader = DataLoader(
            tokenized_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        
        test_loader = DataLoader(
            tokenized_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=default_data_collator
        )
        
        print(f"  - Train samples: {len(tokenized_train)}")
        print(f"  - Test samples: {len(tokenized_test)}")
        
        return train_loader, test_loader
    
    def _prepare_model(self):
        """Load pre-trained model"""
        model = GPT2ForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.to(self.config.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  → Total parameters: {total_params:,}")
        
        return model, self.tokenizer
    
    def _apply_lora(self):
        """Apply LoRA using PEFT"""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
            bias="none"
        )
        
        model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def _prepare_for_dp(self):
        """Make model compatible with Opacus"""
        # Opacus requires models to be compatible with per-sample gradients
        errors = ModuleValidator.validate(self.model, strict=False)
        
        if errors:
            print(f"  - Found {len(errors)} incompatible modules, fixing...")
            self.model = ModuleValidator.fix(self.model)
        else:
            print(f"  - Model is Opacus-compatible")
    
    def _attach_privacy_engine(self):
        """Attach Opacus PrivacyEngine"""
        self.privacy_engine = PrivacyEngine()
        
        # Ghost clipping mode returns (model, optimizer, criterion, data_loader)
        # Standard hooks mode returns (model, optimizer, data_loader)
        result = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config.epochs,
            target_epsilon=self.config.target_epsilon,
            target_delta=self.config.target_delta,
            max_grad_norm=self.config.max_grad_norm,
            grad_sample_mode=self.config.grad_sample_mode,
        )
        
        if self.config.grad_sample_mode == "ghost":
            self.model, self.optimizer, self.criterion, self.train_loader = result
            self.use_ghost_clipping = True
        else:
            self.model, self.optimizer, self.train_loader = result
            self.criterion = None
            self.use_ghost_clipping = False
        
        print(f"  - Target ε: {self.config.target_epsilon}")
        print(f"  - Target δ: {self.config.target_delta:.2e}")
        print(f"  - Noise multiplier: {self.optimizer.noise_multiplier:.4f}")
        print(f"  - Max grad norm (C): {self.config.max_grad_norm}")
        print(f"  - Grad sample mode: {self.config.grad_sample_mode}")
        print(f"  - Ghost clipping: {'ENABLED' if self.use_ghost_clipping else 'DISABLED'}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch.get("labels", batch.get("label")).to(self.config.device)
            
            self.optimizer.zero_grad()
            
            if self.use_ghost_clipping:
                # Ghost clipping: use the criterion wrapper which handles two-pass backward
                # Forward pass - get logits only (no labels to model)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Use the DP criterion wrapper for loss + backward
                # criterion returns DPTensorFastGradientClipping with .backward() and .item()
                loss = self.criterion(outputs.logits, labels)
                loss.backward()  # This performs the two-pass backward for ghost clipping
                loss_value = loss.item()
            else:
                # Standard mode: regular forward + backward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                loss_value = loss.item()
            
            # Optimizer step (Opacus handles clipping and noise)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss_value
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        # Get privacy spent
        epsilon = self.privacy_engine.get_epsilon(self.config.target_delta)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epsilon': epsilon
        }
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                # default_data_collator converts "label" to "labels"
                labels = batch.get("labels", batch.get("label")).to(self.config.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        return accuracy
    
    def train(self):
        """Full training loop"""
        print("\nStarting training...\n")
        
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {metrics['loss']:.4f}")
            print(f"  Train Acc: {metrics['accuracy']:.2f}%")
            print(f"  Privacy Spent (ε): {metrics['epsilon']:.2f}")
            
            # Evaluate every epoch
            test_acc = self.evaluate()
            print(f"  Test Acc: {test_acc:.2f}%")
            print()
        
        # Final privacy accounting
        final_epsilon = self.privacy_engine.get_epsilon(self.config.target_delta)
        print("=" * 50)
        print("Training Complete")
        print(f"Final Privacy: (ε={final_epsilon:.2f}, δ={self.config.target_delta:.2e})")
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    # Test with LoRA (using smaller dataset for faster testing)
    print("\n### BASELINE 1: DP + LoRA ###\n")
    config_lora = BaselineConfig(
        use_lora=True,
        batch_size=16,  # Smaller batch for testing
        epochs=1,
        max_train_samples=100,  # Use smaller subset for quick testing
        max_test_samples=50,
        grad_sample_mode="ghost",  # Enable ghost clipping
    )
    trainer_lora = BaselineDPTrainer(config_lora)
    trainer_lora.setup()
    trainer_lora.train()
    
    # Test without LoRA
    # print("\n\n### BASELINE 2: DP without LoRA ###\n")
    # config_no_lora = BaselineConfig(
    #     use_lora=False,
    #     batch_size=8,
    #     epochs=1
    # )
    # trainer_no_lora = BaselineDPTrainer(config_no_lora)
    # trainer_no_lora.setup()
    # trainer_no_lora.train()