import os
import time
import torch
from datasets import load_from_disk
from transformers import LEDTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments, TrainerCallback
from peft import PeftModel
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
    
    print("Loading Base LED Model...")
    base_model = LEDForConditionalGeneration.from_pretrained(
        model_name,
        gradient_checkpointing=True,
        use_cache=False
    )
    
    rhetorical_sft_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'rhetorical_role', 'final'))
    print(f"Loading SFT LoRA weights for Rhetorical from {rhetorical_sft_weights}...")
    model = PeftModel.from_pretrained(base_model, rhetorical_sft_weights, is_trainable=True)
    model.print_trainable_parameters()
    
    dpo_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'IN-Abs_Rhetorical_DPO'))
    
    print("Loading Rhetorical DPO datasets...")
    train_dataset = load_from_disk(os.path.join(dpo_data_root, "train")).filter(lambda x: len(x['prompt'].split()) < 3000).select(range(1500))
    test_dataset = load_from_disk(os.path.join(dpo_data_root, "test"))
    
    output_weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'dpo_rhetorical'))
    
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
    
    print("Initializing DPOTrainer for Rhetorical...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        beta=0.1, 
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=4096 + 1024,
        max_prompt_length=4096,
        max_target_length=1024,
        callbacks=[CooldownCallback()],
    )
    
    print("Starting Rhetorical DPO Fine-tuning...")
    dpo_trainer.train()
    
    print("Saving final Rhetorical DPO model...")
    final_output_path = os.path.join(output_weights_dir, "final")
    dpo_trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("All done!")

if __name__ == "__main__":
    main()
