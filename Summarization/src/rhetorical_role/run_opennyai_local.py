import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from tqdm import tqdm
import spacy
from opennyai.rhetorical_roles.rhetorical_roles import RhetoricalRolePredictor

# Load Spacy for preprocessing
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

def tag_document_opennyai(predictor, text, filename):
    if not text.strip(): return text
    try:
        doc = nlp(text)
        data = [{
            "file_id": filename,
            "preamble_doc": nlp(""),
            "judgement_doc": doc
        }]
        
        results = predictor(data)
        
        label_map = {
            "PREAMBLE": "Preamble",
            "FAC": "Fact",
            "RLC": "Ruling by Lower Court",
            "ISSUE": "Issue",
            "ARG_PETITIONER": "Argument by Petitioner",
            "ARG_RESPONDENT": "Argument by Respondent",
            "ANALYSIS": "Analysis",
            "STA": "Statute",
            "PRE_RELIED": "Precedent Relied",
            "PRE_NOT_RELIED": "Precedent Not Relied",
            "Ratio of the decision": "Ratio of the Decision",
            "RPC": "Ruling by Present Court",
            "NONE": "Unclassified"
        }
        
        tagged_sentences = []
        for ann in results[0]['annotations']:
            raw_label = ann['labels'][0] if ann.get('labels') else 'NONE'
            sent_text = ann['text'].strip()
            
            if raw_label == 'NONE':
                tagged_sentences.append(f"{sent_text}")
            else:
                english_label = label_map.get(raw_label, raw_label.title())
                tagged_sentences.append(f"{english_label}: {sent_text}")
            
        return " ".join(tagged_sentences)
    except Exception as e:
        import traceback
        print(f"\nError in {filename}")
        print(traceback.format_exc())
        print("---------------------------\n")
        return text

def process_dataset(predictor, input_dir, output_dir):
    judgement_in = os.path.join(input_dir, "judgement")
    summary_in = os.path.join(input_dir, "summary")
    judgement_out = os.path.join(output_dir, "judgement")
    summary_out = os.path.join(output_dir, "summary")
    
    os.makedirs(judgement_out, exist_ok=True)
    os.makedirs(summary_out, exist_ok=True)
    
    if not os.path.exists(judgement_in): return
    
    files = [f for f in os.listdir(judgement_in) if f.endswith(".txt")]
    print(f"Processing {len(files)} files in {input_dir}...")
    
    for filename in tqdm(files):
        if os.path.exists(os.path.join(judgement_out, filename)):
            continue
            
        with open(os.path.join(judgement_in, filename), "r", encoding="utf-8") as f:
            doc_text = f.read().strip()
        with open(os.path.join(summary_in, filename), "r", encoding="utf-8") as f:
            sum_text = f.read().strip()
            
        tagged_doc = tag_document_opennyai(predictor, doc_text, filename)
        
        with open(os.path.join(judgement_out, filename), "w", encoding="utf-8") as f:
            f.write(tagged_doc)
        with open(os.path.join(summary_out, filename), "w", encoding="utf-8") as f:
            f.write(sum_text)

if __name__ == "__main__":
    print("Initializing OpenNYAI pipeline...")
    # Switched back to use_gpu=True for single-threaded processing
    predictor = RhetoricalRolePredictor(use_gpu=True, verbose=False)
    
    raw_data_root = os.path.join("data", "raw", "IN-Abs")
    processed_data_root = os.path.join("data", "processed", "IN-Abs_Rhetorical")
    
    print(f"\n--- Starting OpenNYAI Tagging Pipeline ---")
    start_time = time.time()
    
    try:
        process_dataset(predictor, os.path.join(raw_data_root, "train-data"), os.path.join(processed_data_root, "train-data"))
        process_dataset(predictor, os.path.join(raw_data_root, "test-data"), os.path.join(processed_data_root, "test-data"))
        
        print(f"\nDone! Annotated entire dataset in {time.time() - start_time:.2f} seconds.")
        print("Dataset saved to data/processed/IN-Abs_Rhetorical/")
    except BaseException as e:
        import traceback
        with open("rhetorical_crash_debug.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"Error: Wrote traceback to rhetorical_crash_debug.log")
