import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import random
import logging
from tqdm import tqdm
import multiprocessing

# Setup logging
logging.basicConfig(filename="benchmark_results.log", level=logging.INFO, format="%(message)s")

# Synthetic dataset with on-the-fly preprocessing
class SyntheticTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10000, max_length=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = " ".join(["benchmark"] * random.randint(5, 20))
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return (
            inputs["input_ids"].squeeze(),
            inputs["attention_mask"].squeeze(),
            torch.tensor(1)
        )

def run_benchmark():
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    logging.info(f"Device used: {device}")

    models_to_test = ["bert-base-uncased", "distilbert-base-uncased"]

    for model_name in models_to_test:
        print(f"\n🔍 Benchmarking model: {model_name}")
        logging.info(f"\nBenchmarking model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        dataset = SyntheticTextDataset(tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )

        # Preprocessing benchmark (simulate 10k samples)
        start_pre = time.time()
        for _ in tqdm(range(10000), desc="🧼 Preprocessing"):
            tokenizer("benchmark test sentence", padding="max_length", truncation=True, max_length=128)
        end_pre = time.time()

        # Training benchmark
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model.train()
        start_train = time.time()
        for batch in tqdm(dataloader, desc="🏋️ Training"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        end_train = time.time()

        # Log results
        pre_time = end_pre - start_pre
        train_time = end_train - start_train
        print(f"✅ {model_name} → Preprocessing: {pre_time:.2f}s | Training: {train_time:.2f}s")
        logging.info(f"Preprocessing time (10000 samples): {pre_time:.2f} seconds")
        logging.info(f"Training time (1 epoch, 10000 samples): {train_time:.2f} seconds")

    print("\n📄 Benchmark complete. Results saved to benchmark_results.log")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_benchmark()
