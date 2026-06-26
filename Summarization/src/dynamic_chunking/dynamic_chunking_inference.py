import os
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

class LegalChunkingSummarizer:
    def __init__(self, model_path="allenai/led-base-16384", weights_dir=None, use_tagging=True):
        self.use_tagging = use_tagging
        if self.use_tagging:
            print("Loading OpenNYAI Pipeline...")
            import spacy
            from opennyai.rhetorical_roles.rhetorical_roles import RhetoricalRolePredictor
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                self.nlp.add_pipe("sentencizer")
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                self.nlp.add_pipe("sentencizer")
            self.nyai_pipeline = RhetoricalRolePredictor(use_gpu=True, verbose=False)
        
        print(f"Loading Tokenizer from {model_path}...")
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)
        
        print("Loading LED Model...")
        self.model = LEDForConditionalGeneration.from_pretrained(model_path)
        
        if weights_dir and os.path.exists(weights_dir):
            from peft import PeftModel
            print(f"Loading LoRA weights from {weights_dir}...")
            self.model = PeftModel.from_pretrained(self.model, weights_dir)
            
        self.model.eval()
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.max_tokens = 6144
        self.chunk_size = 5000
        self.overlap = 500

    def _tag_text(self, text):
        if not self.use_tagging: return text
        if not text.strip(): return text
        try:
            doc = self.nlp(text)
            data = [{
                "file_id": "temp",
                "preamble_doc": self.nlp(""),
                "judgement_doc": doc
            }]
            results = self.nyai_pipeline(data)
            
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
            print(f"Error tagging document: {e}")
            return text

    def _generate_summary(self, input_ids):
        # input_ids tensor shape: (1, seq_len)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1 # Global attention on <s> token
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                global_attention_mask=global_attention_mask.to(self.device),
                max_length=1024,
                num_beams=4,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize(self, raw_text: str):
        if self.use_tagging:
            print("Tagging text...")
        tagged_text = self._tag_text(raw_text)
        
        print("Tokenizing text...")
        inputs = self.tokenizer(tagged_text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids[0]
        
        total_tokens = len(input_ids)
        print(f"Total tokens: {total_tokens}")
        
        if total_tokens <= self.max_tokens:
            print("Fitting in context window. Summarizing...")
            return self._generate_summary(inputs.input_ids)
            
        print("Chunking document...")
        # Map Phase
        sub_summaries = []
        start_idx = 0
        
        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0)
            
            print(f"Processing chunk: {start_idx} to {end_idx}...")
            chunk_summary = self._generate_summary(chunk_ids)
            sub_summaries.append(chunk_summary)
            
            start_idx += (self.chunk_size - self.overlap)
            
        # Reduce Phase
        combined_sub_summaries = " ".join(sub_summaries)
        print("Returning combined summaries.")
        return combined_sub_summaries

if __name__ == "__main__":
    print("Dynamic Chunking Pipeline initialized. Import `LegalChunkingSummarizer` to use.")
