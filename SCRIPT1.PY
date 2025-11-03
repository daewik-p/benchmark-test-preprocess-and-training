import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import random

# Detect device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Neural Engine
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU fallback

print(f"Using device: {device}")

# Synthetic dataset
class SyntheticTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000, max_length=128):
        self.samples = []
        for _ in range(num_samples):
            text = " ".join(["benchmark"] * random.randint(5, 20))
            inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            self.samples.append((inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), torch.tensor(1)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Dataset and dataloader
dataset = SyntheticTextDataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Benchmark preprocessing
start_pre = time.time()
_ = [tokenizer("benchmark test sentence", padding="max_length", truncation=True, max_length=128) for _ in range(1000)]
end_pre = time.time()

# Benchmark training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
start_train = time.time()
for batch in dataloader:
    input_ids, attention_mask, labels = [x.to(device) for x in batch]
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
end_train = time.time()

# Results
print(f"\n--- Benchmark Results ---")
print(f"Preprocessing time (1000 samples): {end_pre - start_pre:.2f} seconds")
print(f"Training time (1 epoch, 1000 samples): {end_train - start_train:.2f} seconds")
print(f"Device used: {device}")
