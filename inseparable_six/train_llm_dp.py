import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm

# -------------------------
# 1. Configuration
# -------------------------
model_name = "gpt2"  # or a smaller LLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layer-wise DP hyperparameters
layer_clipping_norms = {}       # e.g., {"transformer.h.0": 0.5, ...}
layer_noise_multipliers = {}    # e.g., {"transformer.h.0": 1.0, ...}
default_clip = 1.0
default_noise = 1.0

batch_size = 4
num_epochs = 3
lr = 5e-5

# -------------------------
# 2. Load model & tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=lr)

# -------------------------
# 3. Dataset
# -------------------------
# Replace with your sensitive dataset
texts = [
    "password: hunter2", 
    "secret token: abc123", 
    "regular sentence for training"
]

encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# 4. Optimizer
# -------------------------
optimizer = AdamW(model.parameters(), lr=lr)

# -------------------------
# 5. Training loop with layer-wise DP
# -------------------------
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        input_ids, attention_mask = [x.to(device) for x in batch]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        
        # -------- Layer-wise DP: Clip + Add noise per parameter --------
        for name, param in model.named_parameters():
            if param.grad is not None:
                clip_value = layer_clipping_norms.get(name, default_clip)
                noise_multiplier = layer_noise_multipliers.get(name, default_noise)
                
                # Clip gradient
                grad_norm = param.grad.norm()
                param.grad.data *= min(1.0, clip_value / (grad_norm + 1e-6))
                
                # Add Gaussian noise
                noise = torch.randn_like(param.grad) * (noise_multiplier * clip_value)
                param.grad.data += noise
        
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} done")

print("Layer-wise DP fine-tuning complete!")
