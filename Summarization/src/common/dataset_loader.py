import os
from datasets import Dataset

def load_in_abs_split(split_dir):
    judgement_dir = os.path.join(split_dir, "judgement")
    summary_dir = os.path.join(split_dir, "summary")
    
    data = {"id": [], "document": [], "summary": []}
    
    for filename in os.listdir(judgement_dir):
        if not filename.endswith(".txt"):
            continue
            
        judgement_path = os.path.join(judgement_dir, filename)
        summary_path = os.path.join(summary_dir, filename)
        
        if not os.path.exists(summary_path):
            continue
            
        with open(judgement_path, "r", encoding="utf-8") as f:
            doc_text = f.read().strip()
            
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_text = f.read().strip()
            
        data["id"].append(filename.replace(".txt", ""))
        data["document"].append(doc_text)
        data["summary"].append(sum_text)
        
    return Dataset.from_dict(data)

def get_in_abs_datasets(dataset_root="data/raw/IN-Abs"):
    train_dir = os.path.join(dataset_root, "train-data")
    test_dir = os.path.join(dataset_root, "test-data")
    
    print(f"Loading training data from {train_dir}...")
    train_dataset = load_in_abs_split(train_dir)
    
    print(f"Loading test data from {test_dir}...")
    test_dataset = load_in_abs_split(test_dir)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples.")
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_ds, test_ds = get_in_abs_datasets()
    print("Sample Train Document:", train_ds[0]["document"][:200])
    print("Sample Train Summary:", train_ds[0]["summary"][:200])
