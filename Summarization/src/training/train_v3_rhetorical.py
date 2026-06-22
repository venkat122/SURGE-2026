import torch
import sys
import os
from transformers import LEDTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_processing.dataset_loader import get_in_abs_datasets

def main():
    # Load the V3 Tagged Dataset
    v3_dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_V3'))
    
    print(f"Loading Phase 3 Tagged Dataset from: {v3_dataset_root}")
    train_dataset, test_dataset = get_in_abs_datasets(dataset_root=v3_dataset_root)
    
    # We use allenai/led-base-16384 which natively handles 16k tokens
    model_name = "allenai/led-base-16384"
    print(f"Loading Tokenizer and Model: {model_name}...")
    
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    
    # Cap encoder length to 8192 to save memory
    # Expanded decoder max length to 1024 for more detailed abstractive summaries
    encoder_max_length = 8192
    decoder_max_length = 1024
    
    def process_data_to_model_inputs(batch):
        # Tokenize the input and output texts
        inputs = tokenizer(
            batch["document"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        
        # LED requires global attention on specific tokens (usually the first <s> token).
        global_attention_mask = len(inputs.input_ids) * [[0 for _ in range(len(inputs.input_ids[0]))]]
        for i in range(len(inputs.input_ids)):
            global_attention_mask[i][0] = 1
        batch["global_attention_mask"] = global_attention_mask

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in outputs.input_ids
        ]

        return batch

    print("Tokenizing tagged datasets... this may take some RAM")
    train_dataset = train_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=2, 
        remove_columns=["id", "document", "summary"]
    )
    
    test_dataset = test_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=2, 
        remove_columns=["id", "document", "summary"]
    )
    
    print("Tokenization complete! Setting format...")
    
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
    )
    
    print("Initializing LED Model...")
    model = LEDForConditionalGeneration.from_pretrained(
        model_name, 
        gradient_checkpointing=True, 
        use_cache=False 
    )

    # ==========================================
    # COVERAGE PENALTY (Inference / Generation limits)
    # ==========================================
    model.config.repetition_penalty = 1.5
    model.config.no_repeat_ngram_size = 3
    model.config.max_length = 1024
    # ==========================================

    # ==========================================
    # EXPANDED LoRA CONFIGURATION
    # ==========================================
    print("Applying Expanded LoRA to the model...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,                                       
        lora_alpha=64,                              
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"] 
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    # ==========================================

    # Output to V3 weights folder
    output_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'v3_rhetorical'))

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_weights_dir,
        predict_with_generate=True,
        generation_max_length=1024,
        evaluation_strategy="steps",
        per_device_train_batch_size=1,        
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,        
        fp16=True,                            
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        save_total_limit=2,                   
        num_train_epochs=3,                   
        learning_rate=3e-4,
        lr_scheduler_type="cosine",           
        warmup_ratio=0.1,                     
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("\n--- Phase 3 Model Preparation Complete ---")
    print("Starting LED V3 fine-tuning on Rhetorical-Tagged Dataset...")
    
    trainer.train()

if __name__ == "__main__":
    main()
