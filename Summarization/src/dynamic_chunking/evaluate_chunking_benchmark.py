import torch
import evaluate
import json
import os
import sys
from transformers import LEDTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.dataset_loader import get_in_abs_datasets
from src.dynamic_chunking.dynamic_chunking_inference import LegalChunkingSummarizer

def main():
    print("Loading IN-Abs test datasets...")
    raw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'IN-Abs'))
    tagged_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_Rhetorical'))
    
    _, test_raw = get_in_abs_datasets(dataset_root=raw_root)
    _, test_tagged = get_in_abs_datasets(dataset_root=tagged_root)

    print("Tokenizing to find the top 25 longest documents...")
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    
    docs_with_lengths = []
    for i, example in enumerate(test_raw):
        tokens = tokenizer(example['document'], truncation=False)['input_ids']
        docs_with_lengths.append({
            'id': example['id'],
            'length': len(tokens),
            'raw_doc': example['document'],
            'summary': example['summary']
        })

    # Sort by length descending and take top 15
    docs_with_lengths.sort(key=lambda x: x['length'], reverse=True)
    top_15_docs = docs_with_lengths[:15]
    print(f"Longest doc length: {top_15_docs[0]['length']} tokens")
    print(f"15th doc length: {top_15_docs[-1]['length']} tokens")
    
    # Map tagged docs by ID
    tagged_map = {ex['id']: ex['document'] for ex in test_tagged}

    models_to_test = [
        {"name": "Optimized Baseline", "weights": "optimized_baseline_small/final", "needs_tagged_text": False},
        {"name": "Rhetorical Model", "weights": "rhetorical_role/final", "needs_tagged_text": True},
        {"name": "DPO Baseline", "weights": "dpo_baseline/final", "needs_tagged_text": False},
        {"name": "DPO Rhetorical", "weights": "dpo_rhetorical/final", "needs_tagged_text": True},
    ]

    base_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights'))
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chunking_empirical_results.json'))
    
    print("Loading metrics...")
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    results = {}
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            results = json.load(f)

    for model_info in models_to_test:
        name = model_info["name"]
        
        if name in results and "Chunking" in results[name] and "Standard" in results[name]:
            print(f"Skipping {name}, already fully evaluated.")
            continue
            
        weights_path = os.path.join(base_weights_dir, model_info["weights"])
        needs_tagged_text = model_info["needs_tagged_text"]
        
        if not os.path.exists(weights_path):
            print(f"Skipping {name}, weights not found at {weights_path}")
            continue

        print(f"\n======================================")
        print(f"Evaluating: {name}")
        print(f"======================================")

        # We pass use_tagging=False ALWAYS to prevent spacy imports crashing the script.
        # We manually feed the pre-tagged text instead!
        summarizer = LegalChunkingSummarizer(weights_dir=weights_path, use_tagging=False)
        
        if name not in results:
            results[name] = {}

        for mode in ["Standard", "Chunking"]:
            if mode in results[name]:
                continue
                
            print(f"\n--- Running in {mode} Mode ---")
            predictions = []
            references = []

            for i, doc_info in enumerate(top_15_docs):
                if needs_tagged_text:
                    text_to_feed = tagged_map[doc_info['id']]
                else:
                    text_to_feed = doc_info['raw_doc']
                    
                ref_summary = doc_info['summary']
                
                print(f"Doc {i+1}/15 (Length: {doc_info['length']} tokens) - {mode} Mode")
                if mode == "Standard":
                    # Force model to not map-reduce by feeding it truncated text
                    original_max = summarizer.max_tokens
                    summarizer.max_tokens = 999999 
                    
                    tokens = summarizer.tokenizer(text_to_feed, truncation=True, max_length=6144)
                    truncated_text = summarizer.tokenizer.decode(tokens.input_ids, skip_special_tokens=True)
                    
                    pred = summarizer.summarize(truncated_text)
                    summarizer.max_tokens = original_max
                else:
                    pred = summarizer.summarize(text_to_feed)

                predictions.append(pred)
                references.append(ref_summary)

            print(f"Computing metrics for {name} ({mode})...")
            r_scores = rouge.compute(predictions=predictions, references=references)
            b_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
            
            b_f1 = sum(b_scores['f1']) / len(b_scores['f1'])
            
            results[name][mode] = {
                "ROUGE-1": r_scores["rouge1"],
                "ROUGE-2": r_scores["rouge2"],
                "ROUGE-L": r_scores["rougeL"],
                "BERTScore F1": b_f1
            }
            
            print(results[name][mode])
            
            # Save incrementally
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=4)

    print(f"\nBenchmark complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()
