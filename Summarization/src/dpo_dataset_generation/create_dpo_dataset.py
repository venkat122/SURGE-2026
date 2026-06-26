import os
import json
import random
import re
from datasets import Dataset

def inject_hallucination(text):
    # Basic rule-based hallucination injection
    hallucinated = text
    
    # Swap Petitioner and Respondent
    hallucinated = re.sub(r'\b(petitioner|appellant)\b', 'TEMP_RESP', hallucinated, flags=re.IGNORECASE)
    hallucinated = re.sub(r'\b(respondent|defendant)\b', 'petitioner', hallucinated, flags=re.IGNORECASE)
    hallucinated = hallucinated.replace('TEMP_RESP', 'respondent')
    
    # Negate outcomes
    replacements = {
        "appeal is allowed": "appeal is dismissed",
        "appeal is dismissed": "appeal is allowed",
        "is guilty": "is not guilty",
        "conviction is upheld": "conviction is overturned",
        "conviction is overturned": "conviction is upheld"
    }
    
    for old, new in replacements.items():
        hallucinated = re.sub(rf'\b{old}\b', new, hallucinated, flags=re.IGNORECASE)
        
    return hallucinated

def build_dpo_dataset(input_dir, output_path):
    judgement_dir = os.path.join(input_dir, "judgement")
    summary_dir = os.path.join(input_dir, "summary")
    
    if not os.path.exists(judgement_dir):
        print(f"Data directory not found: {input_dir}")
        return
        
    data = {"prompt": [], "chosen": [], "rejected": []}
    
    files = [f for f in os.listdir(judgement_dir) if f.endswith(".txt")]
    for filename in files:
        with open(os.path.join(judgement_dir, filename), "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        with open(os.path.join(summary_dir, filename), "r", encoding="utf-8") as f:
            chosen = f.read().strip()
            
        rejected = inject_hallucination(chosen)
        
        # Only add if the hallucination actually changed something
        if rejected != chosen:
            data["prompt"].append(prompt)
            data["chosen"].append(chosen)
            data["rejected"].append(rejected)
            
    ds = Dataset.from_dict(data)
    ds.save_to_disk(output_path)
    print(f"Saved {len(ds)} DPO preference pairs to {output_path}")

if __name__ == "__main__":
    # Optimized Dataset (Raw Text)
    optimized_raw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'IN-Abs'))
    dpo_baseline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_DPO_Baseline'))
    
    # Rhetorical Dataset (Tagged Text)
    rhetorical_raw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_Rhetorical'))
    dpo_rhetorical_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_Rhetorical_DPO'))
    
    print("Building Optimized DPO Dataset (Raw Text)...")
    os.makedirs(dpo_baseline_root, exist_ok=True)
    build_dpo_dataset(os.path.join(optimized_raw_root, "train-data"), os.path.join(dpo_baseline_root, "train"))
    build_dpo_dataset(os.path.join(optimized_raw_root, "test-data"), os.path.join(dpo_baseline_root, "test"))
    
    print("\nBuilding Rhetorical DPO Dataset (Tagged Text)...")
    os.makedirs(dpo_rhetorical_root, exist_ok=True)
    build_dpo_dataset(os.path.join(rhetorical_raw_root, "train-data"), os.path.join(dpo_rhetorical_root, "train"))
    build_dpo_dataset(os.path.join(rhetorical_raw_root, "test-data"), os.path.join(dpo_rhetorical_root, "test"))
