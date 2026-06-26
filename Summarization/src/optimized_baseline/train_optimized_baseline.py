import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
import time

class CooldownCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # Triggered immediately after the checkpoint is saved to disk
        print(f"\n[CooldownCallback] Checkpoint saved at step {state.global_step}. Pausing for 30 minutes to cool down the laptop GPU...")
        time.sleep(1800) # 30 minutes
        print("[CooldownCallback] Cooldown finished! Resuming training...\n")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.dataset_loader import get_in_abs_datasets

def main():
    # Load dataset
    train_dataset, test_dataset = get_in_abs_datasets()
    
    # We use allenai/led-base-16384 which natively handles 16k tokens
    model_name = "allenai/led-base-16384"
    print(f"Loading Tokenizer and Model: {model_name}...")
    
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    
    # Cap encoder length to 8192 to save memory
    # Expanded decoder max length to 1024 for more detailed abstractive summaries
    encoder_max_length = 6144
    decoder_max_length = 1024
    
    from transformers import DataCollatorForSeq2Seq

    def process_data_to_model_inputs(batch):
        # Tokenize dynamically (no static padding here to save massive VRAM on short documents)
        inputs = tokenizer(
            batch["document"],
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["summary"],
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids
        # LED automatically applies global attention to the first <s> token if global_attention_mask is omitted.

        return batch

    print("Tokenizing datasets... processing")
    train_dataset = train_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=2, 
        writer_batch_size=10,
        keep_in_memory=False,
        remove_columns=["id", "document", "summary"]
    )
    
    test_dataset = test_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=2, 
        writer_batch_size=10,
        keep_in_memory=False,
        remove_columns=["id", "document", "summary"]
    )
    
    print("Tokenization complete! Setting format...")
    
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
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
    # These settings enforce a penalty against generating the same n-grams or tokens
    # effectively acting as a coverage penalty during evaluation.
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
        r=16,                                       # DECREASED Rank
        lora_alpha=32,                              # DECREASED Scaling factor
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"] # TARGETING QKV ONLY
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    # ==========================================
    
    output_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'optimized_baseline_small'))

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_weights_dir,
        predict_with_generate=False,
        generation_max_length=1024,
        evaluation_strategy="steps",
        per_device_train_batch_size=1,        
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,        
        dataloader_num_workers=0,             # CHANGED TO 0 TO PREVENT RAM EXHAUSTION ON WINDOWS
        bf16=True,
        tf32=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        save_total_limit=2,                   
        num_train_epochs=2,                   
        learning_rate=3e-4,
        lr_scheduler_type="cosine",           # COSINE DECAY added
        warmup_ratio=0.1,                     # 10% WARMUP added
        optim="adamw_torch_fused",            # FUSED ADAMW FOR FASTER TRAINING ON L4 GPU
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[CooldownCallback()],
    )

    print("\n--- Model Preparation Complete ---")
    print("Starting training...")
    
    # Train Model (fresh restart)
    trainer.train(resume_from_checkpoint=False)

    print("Training Complete! Saving final model weights...")
    final_output_path = os.path.join(output_weights_dir, "final")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("All done!")

if __name__ == "__main__":
    main()
