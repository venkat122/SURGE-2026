import os
import re
from tqdm import tqdm

def tag_sentence(sentence):
    sentence_lower = sentence.lower()
    
    # Precedent (Citations to other cases)
    if re.search(r'\b(v\.|vs\.|air|scc|scr|supra)\b', sentence_lower) or "relied on the decision" in sentence_lower:
        return "[Precedent] " + sentence
        
    # Statute (Laws and Acts)
    if re.search(r'\b(section|article|act|clause|ipc|crpc)\b', sentence_lower):
        return "[Statute] " + sentence
        
    # Arguments (What the lawyers say)
    if re.search(r'\b(contends|submits|argues|learned counsel|appellant claims|respondent submitted)\b', sentence_lower):
        return "[Argument] " + sentence
        
    # Ruling / Final Decision
    if re.search(r'\b(appeal is dismissed|appeal is allowed|decree is passed|order accordingly|disposed of)\b', sentence_lower):
        return "[Ruling] " + sentence
        
    # Issues
    if sentence_lower.startswith("whether") or "the question before us" in sentence_lower or "the issue is" in sentence_lower:
        return "[Issue] " + sentence
        
    # Facts
    if "briefly the facts" in sentence_lower or "factual matrix" in sentence_lower or "incident occurred" in sentence_lower:
        return "[Fact] " + sentence
        
    # Default (No specific tag, treated as general Analysis/Ratio)
    return sentence

def process_document(text):
    # Very basic sentence splitting for speed.
    # In a production environment, use spaCy: nlp(text).sents
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    
    tagged_sentences = []
    for s in sentences:
        # Add period back if missing
        if not s.endswith('.'):
            s += '.'
        tagged_sentences.append(tag_sentence(s))
        
    return ' '.join(tagged_sentences)

def process_dataset(input_dir, output_dir):
    judgement_in_dir = os.path.join(input_dir, "judgement")
    summary_in_dir = os.path.join(input_dir, "summary")
    
    judgement_out_dir = os.path.join(output_dir, "judgement")
    summary_out_dir = os.path.join(output_dir, "summary")
    
    os.makedirs(judgement_out_dir, exist_ok=True)
    os.makedirs(summary_out_dir, exist_ok=True)
    
    if not os.path.exists(judgement_in_dir):
        return
        
    files = [f for f in os.listdir(judgement_in_dir) if f.endswith(".txt")]
    
    print(f"Processing {len(files)} files in {input_dir}...")
    for filename in tqdm(files):
        # 1. Read input files
        judgement_path = os.path.join(judgement_in_dir, filename)
        summary_path = os.path.join(summary_in_dir, filename)
        
        if not os.path.exists(summary_path):
            continue
            
        with open(judgement_path, "r", encoding="utf-8") as f:
            doc_text = f.read().strip()
            
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_text = f.read().strip()
            
        # 2. Tag the document
        tagged_doc = process_document(doc_text)
        
        # 3. Save to output directory
        with open(os.path.join(judgement_out_dir, filename), "w", encoding="utf-8") as f:
            f.write(tagged_doc)
            
        with open(os.path.join(summary_out_dir, filename), "w", encoding="utf-8") as f:
            f.write(sum_text)

def main():
    raw_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'IN-Abs'))
    processed_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_V3'))
    
    print(f"Input Directory: {raw_data_root}")
    print(f"Output Directory: {processed_data_root}")
    
    train_in = os.path.join(raw_data_root, "train-data")
    train_out = os.path.join(processed_data_root, "train-data")
    process_dataset(train_in, train_out)
    
    test_in = os.path.join(raw_data_root, "test-data")
    test_out = os.path.join(processed_data_root, "test-data")
    process_dataset(test_in, test_out)
    
    print("Phase 3 Rhetorical Role Tagging Complete!")

if __name__ == "__main__":
    main()
