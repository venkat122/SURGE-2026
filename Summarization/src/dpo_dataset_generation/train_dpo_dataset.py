import os
import time
import torch
from datasets import load_from_disk
from transformers import LEDTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer

class CooldownCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"\n[CooldownCallback] Checkpoint saved at step {state.global_step}. Pausing for 30 minutes to cool down the laptop GPU...")
        time.sleep(1800)
        print("[CooldownCallback] Cooldown finished! Resuming training...\n")

def main():
    model_name = "allenai/led-base-16384"
    print(f"Loading Tokenizer: {model_name}...")
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    
    print("Loading LED Model...")
    model = LEDForConditionalGeneration.from_pretrained(
        model_name,
        gradient_checkpointing=True,
        use_cache=False
    )
    
    print("Applying LoRA to the model for DPO...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"] 
    )
    model = get_peft_model(model, lora_config)
    
    dpo_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_DPO_Generated'))
    
    print("Loading DPO datasets...")
    train_dataset = load_from_disk(os.path.join(dpo_data_root, "train"))
    test_dataset = load_from_disk(os.path.join(dpo_data_root, "test"))
    
    output_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'dpo_dataset_generation'))
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_weights_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        learning_rate=1e-5, 
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        save_steps=500,
        logging_steps=100,
        bf16=True,
        tf32=True,
        remove_unused_columns=False, 
    )
    
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        beta=0.1, 
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=6144 + 1024,
        max_prompt_length=6144,
        callbacks=[CooldownCallback()],
    )
    
    print("Starting DPO Fine-tuning to eliminate hallucinations...")
    dpo_trainer.train()
    
    print("Saving final DPO model...")
    dpo_trainer.save_model()

if __name__ == "__main__":
    main()
