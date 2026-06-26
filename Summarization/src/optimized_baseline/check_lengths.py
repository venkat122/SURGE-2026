import sys
import os
from transformers import LEDTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.dataset_loader import get_in_abs_datasets

def main():
    print("Loading dataset...")
    train_dataset, test_dataset = get_in_abs_datasets()
    print("Loading tokenizer...")
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    
    total = len(train_dataset)
    over_256 = 0
    over_512 = 0
    over_768 = 0
    over_1024 = 0
    
    print(f"Tokenizing {total} summaries to check lengths...")
    
    for doc in tqdm(train_dataset["summary"]):
        # Tokenize without truncation to get the absolute true token length
        length = len(tokenizer(doc, truncation=False)["input_ids"])
        
        if length > 256: over_256 += 1
        if length > 512: over_512 += 1
        if length > 768: over_768 += 1
        if length > 1024: over_1024 += 1

    print("\n--- Summary Length Statistics (Tokens) ---")
    print(f"Total Summaries: {total}")
    print(f"Over 256 tokens: {over_256} ({(over_256/total)*100:.1f}%)")
    print(f"Over 512 tokens: {over_512} ({(over_512/total)*100:.1f}%)")
    print(f"Over 768 tokens: {over_768} ({(over_768/total)*100:.1f}%)")
    print(f"Over 1024 tokens: {over_1024} ({(over_1024/total)*100:.1f}%)")

if __name__ == "__main__":
    main()
