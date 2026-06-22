import os
from transformers import AutoTokenizer
from led_dataset import get_in_abs_datasets
import numpy as np

def main():
    print("Loading datasets...")
    train_ds, test_ds = get_in_abs_datasets()
    
    # Combine the documents to get a complete picture of the dataset lengths
    all_documents = list(train_ds["document"]) + list(test_ds["document"])
    
    print(f"Loaded {len(all_documents)} total documents.")
    print("Approximating token lengths (word count * 1.3) since HuggingFace download is blocked...")
    lengths = []
    
    for i, doc in enumerate(all_documents):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{len(all_documents)} documents...")
        
        # Approximate tokens by splitting by whitespace and multiplying by 1.3
        words = len(doc.split())
        approx_tokens = int(words * 1.3)
        lengths.append(approx_tokens)
    
    lengths = np.array(lengths)
    
    print("\n--- IN-Abs Dataset Token Length Statistics ---")
    print(f"Total Documents: {len(lengths)}")
    print(f"Minimum Length:  {np.min(lengths):,} tokens")
    print(f"Maximum Length:  {np.max(lengths):,} tokens")
    print(f"Average Length:  {np.mean(lengths):,.0f} tokens")
    print(f"Median Length:   {np.median(lengths):,.0f} tokens")
    
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"{p}th Percentile: {np.percentile(lengths, p):,.0f} tokens")
        
    exceed_1024 = np.sum(lengths > 1024)
    exceed_4096 = np.sum(lengths > 4096)
    exceed_8192 = np.sum(lengths > 8192)
    exceed_16384 = np.sum(lengths > 16384)
    
    print("\n--- Token Thresholds ---")
    print(f"Documents exceeding 1,024 tokens (T5 limit): {exceed_1024} ({exceed_1024/len(lengths)*100:.1f}%)")
    print(f"Documents exceeding 4,096 tokens:           {exceed_4096} ({exceed_4096/len(lengths)*100:.1f}%)")
    print(f"Documents exceeding 8,192 tokens:           {exceed_8192} ({exceed_8192/len(lengths)*100:.1f}%)")
    print(f"Documents exceeding 16,384 tokens (LED limit):{exceed_16384} ({exceed_16384/len(lengths)*100:.1f}%)")

if __name__ == "__main__":
    main()
