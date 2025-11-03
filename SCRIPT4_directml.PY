import time
import torch_directml
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
    # Use DirectML device
    device = torch_directml.device()
    print(f"Using device: {device}")
    logging.info(f"Device used: DirectML")

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
            pin_memory=False,
            persistent_workers=True
        )

        # Preprocessing benchmark
        start_pre = time.time()
        for _ in tqdm(range(10000), desc="🧼 Preprocessing"):
            tokenizer("benchmark test sentence", padding="max_length", truncation=True, max_length=128)
        end_pre = time.time()

        # Inference benchmark (no training with DirectML)
        model.eval()
        start_infer = time.time()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="🤖 Inference"):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end_infer = time.time()

        # Log results
        pre_time = end_pre - start_pre
        infer_time = end_infer - start_infer
        print(f"✅ {model_name} → Preprocessing: {pre_time:.2f}s | Inference: {infer_time:.2f}s")
        logging.info(f"Preprocessing time (10000 samples): {pre_time:.2f} seconds")
        logging.info(f"Inference time (10000 samples): {infer_time:.2f} seconds")

    print("\n📄 Benchmark complete. Results saved to benchmark_results.log")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_benchmark()
