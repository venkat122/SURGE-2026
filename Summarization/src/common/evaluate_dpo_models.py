import torch
import evaluate
from transformers import LEDTokenizer, LEDForConditionalGeneration
from peft import PeftModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.dataset_loader import get_in_abs_datasets

def main():
    print("Loading test datasets...")
    # Load Optimized (Baseline) and Rhetorical (Rhetorical) datasets
    optimized_dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'IN-Abs'))
    rhetorical_dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_Rhetorical'))
    
    _, test_dataset_optimized = get_in_abs_datasets(dataset_root=optimized_dataset_root)
    _, test_dataset_rhetorical = get_in_abs_datasets(dataset_root=rhetorical_dataset_root)

    # Shuffle and select a subset of 30 documents for fast evaluation
    subset_size = 30
    test_dataset_optimized = test_dataset_optimized.shuffle(seed=42).select(range(subset_size))
    test_dataset_rhetorical = test_dataset_rhetorical.shuffle(seed=42).select(range(subset_size))

    model_name = "allenai/led-base-16384"
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    base_model = LEDForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    base_model.eval()

    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    def evaluate_model(weights_dir, test_dataset, model_label):
        print(f"\n--- Evaluating {model_label} ---")
        if not os.path.exists(weights_dir):
            print(f"Weights dir not found: {weights_dir}")
            return None

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, weights_dir)
        model.to(device)
        model.eval()

        predictions = []
        references = []

        for i, example in enumerate(test_dataset):
            print(f"\rGenerating summary {i+1}/{subset_size}...", end="")
            inputs = tokenizer(example["document"], truncation=True, max_length=6144, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    global_attention_mask=global_attention_mask,
                    max_length=1024,
                    num_beams=4,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3
                )
            
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append(example["summary"])

        print("\nCalculating metrics...")
        rouge_results = rouge.compute(predictions=predictions, references=references)
        
        # Calculate BERTScore
        bert_results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
        avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])

        print(f"\n{model_label} Results:")
        print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"BERTScore F1: {avg_bert_f1:.4f}")

        # Unload LoRA to avoid memory issues when loading the next one
        model.cpu()
        del model
        torch.cuda.empty_cache()

        return {"rouge": rouge_results, "bertscore": avg_bert_f1}

    optimized_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'dpo_baseline', 'final'))
    rhetorical_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'dpo_rhetorical', 'final'))

    evaluate_model(optimized_weights, test_dataset_optimized, "Optimized DPO")
    evaluate_model(rhetorical_weights, test_dataset_rhetorical, "Rhetorical DPO")

if __name__ == "__main__":
    main()
