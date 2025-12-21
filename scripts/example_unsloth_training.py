#!/usr/bin/env python3
"""
Example Unsloth Training Script
Demonstrates basic LoRA fine-tuning with Unsloth on RTX 5060 Ti (16GB)

This is a template/example - modify for your specific use case.
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def main():
    print("=" * 80)
    print("Unsloth Training Example")
    print("=" * 80)

    # Configuration
    max_seq_length = 2048  # Can increase to 4096 if you have enough VRAM
    model_name = "unsloth/mistral-7b-bnb-4bit"  # 4-bit quantized model

    # For RTX 5060 Ti (16GB), these settings should work well
    batch_size = 2
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Step 1: Load model with Unsloth
    print("\n[1/5] Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    print("✓ Model loaded")

    # Step 2: Add LoRA adapters
    print("\n[2/5] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank (8, 16, 32, or 64)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("✓ LoRA adapters added")

    # Step 3: Load dataset
    print("\n[3/5] Loading dataset...")
    # Example: Using a simple instruction dataset
    # Replace with your own dataset!
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # Small subset for demo
    print(f"✓ Loaded {len(dataset)} examples")

    # Format dataset for instruction tuning
    def format_instruction(example):
        if example.get("input", ""):
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    # Step 4: Configure training
    print("\n[4/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir="D:/training/outputs/unsloth_demo",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=100,  # Short demo - increase for real training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=50,
        logging_dir="D:/training/logs",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Will be created by formatting_func
        formatting_func=format_instruction,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can enable for better efficiency
        args=training_args,
    )
    print("✓ Trainer configured")

    # Step 5: Train
    print("\n[5/5] Starting training...")
    print("=" * 80)

    try:
        trainer_stats = trainer.train()
        print("\n" + "=" * 80)
        print("✓ Training completed!")
        print(f"  Total time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
        print(f"  Samples/second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return 1

    # Save model
    print("\n[6/6] Saving model...")
    output_dir = "D:/training/models/unsloth_demo_final"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to: {output_dir}")

    # Optional: Save merged model (LoRA weights merged with base model)
    print("\nSaving merged model...")
    model.save_pretrained_merged(
        "D:/training/models/unsloth_demo_merged",
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for smaller size
    )
    print("✓ Merged model saved")

    print("\n" + "=" * 80)
    print("Training Example Completed Successfully!")
    print("=" * 80)

    print("\nNext steps:")
    print(f"1. Test the model: Load from {output_dir}")
    print("2. Modify this script for your dataset")
    print("3. Increase max_steps for real training")
    print("4. Monitor with: nvidia-smi -l 1")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
