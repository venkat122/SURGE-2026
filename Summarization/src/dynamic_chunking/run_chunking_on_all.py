import os
import sys
import gc
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dynamic_chunking.dynamic_chunking_inference import LegalChunkingSummarizer

def main():
    print("Loading datasets to find a long document...")
    # Load raw dataset to get a long document
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'IN-Abs', 'test-data', 'judgement'))
    
    # Pick a long document
    files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
    longest_file = None
    max_len = 0
    raw_text = ""
    for f in files[:50]:
        with open(os.path.join(dataset_path, f), 'r', encoding='utf-8') as file:
            text = file.read()
            if len(text) > max_len:
                max_len = len(text)
                longest_file = f
                raw_text = text
                
    print(f"Selected long document {longest_file} with length {max_len} chars.")

    models_to_test = [
        {"name": "Optimized Baseline", "weights": "optimized_baseline_small/final", "use_tagging": False},
        {"name": "Rhetorical Rhetorical", "weights": "rhetorical_role/final", "use_tagging": True},
        {"name": "Optimized DPO", "weights": "dpo_baseline/final", "use_tagging": False},
        {"name": "Rhetorical DPO", "weights": "dpo_rhetorical/final", "use_tagging": True},
    ]

    base_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights'))
    output_file = r"C:\Users\samba\.gemini\antigravity\brain\65b10c10-535e-480c-9673-8ae8ea226dea\chunking_results.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dynamic Chunking Results on Long Document\n\n")
        f.write(f"**Document**: `{longest_file}`\n")
        f.write(f"**Length**: {max_len} characters\n\n")

    for model_info in models_to_test:
        name = model_info["name"]
        weights_path = os.path.join(base_weights_dir, model_info["weights"])
        use_tagging = model_info["use_tagging"]
        
        if not os.path.exists(weights_path):
            print(f"Skipping {name}, weights not found at {weights_path}")
            continue
            
        print(f"\n--- Testing {name} ---")
        summarizer = LegalChunkingSummarizer(weights_dir=weights_path, use_tagging=use_tagging)
        
        try:
            summary = summarizer.summarize(raw_text)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"## {name}\n")
                f.write(summary + "\n\n")
        except Exception as e:
            print(f"Error during {name}: {e}")
            
        # Free memory
        del summarizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
