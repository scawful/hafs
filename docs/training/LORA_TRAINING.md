# LoRA Training Guide

## Overview

LoRA (Low-Rank Adaptation) is parameter-efficient fine-tuning that trains small adapter matrices instead of the full model. This allows training 14B-32B models on your RTX 4070 Ti SUPER (16GB VRAM).

## How LoRA Works

Instead of updating all model parameters (billions), LoRA:
1. Freezes the base model weights
2. Injects trainable rank decomposition matrices into each layer
3. Only trains these small adapter matrices (few million parameters)
4. Merges adapters back into base model after training

**Memory Savings:**
- Full fine-tune 14B model: ~80GB VRAM
- LoRA 14B model: ~12-16GB VRAM (fits your GPU!)
- Training speed: 4-8x faster

## Architecture

```
Base Model (frozen)     LoRA Adapters (trainable)
┌─────────────┐        ┌──────────────┐
│   Qwen2.5   │   +    │  LoRA-A (r×d)│
│ 14B params  │        │  LoRA-B (d×r)│
│  (frozen)   │        │   ~4M params │
└─────────────┘        └──────────────┘
```

Where:
- r = rank (typically 8, 16, 32, 64)
- d = hidden dimension (model-specific)
- Higher r = more capacity, more VRAM

## Training Configuration

### Recommended Settings (medical-mechanica)

```python
# Base model
base_model = "Qwen/Qwen2.5-Coder-14B-Instruct"

# LoRA config
lora_r = 64              # Rank (try 32, 64, 128)
lora_alpha = 128         # Scaling factor (typically 2×r)
lora_dropout = 0.05      # Prevent overfitting
target_modules = [       # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # FFN
]

# Training hyperparams
batch_size = 4           # Per-device batch size
gradient_accumulation = 4  # Effective batch = 16
learning_rate = 2e-4     # Higher than full fine-tune
num_epochs = 3           # 1-3 epochs typical
warmup_steps = 100       # LR warmup
weight_decay = 0.01      # Regularization

# Optimizations
fp16 = True              # Mixed precision (2x memory savings)
gradient_checkpointing = True  # Trade compute for memory
```

### Memory Requirements

| Model Size | LoRA Rank | VRAM (fp16) | Batch Size | Training Speed |
|------------|-----------|-------------|------------|----------------|
| 7B         | 32        | 8GB         | 4          | ~1000 tok/s   |
| 7B         | 64        | 10GB        | 4          | ~800 tok/s    |
| 14B        | 32        | 12GB        | 2          | ~500 tok/s    |
| 14B        | 64        | 16GB        | 2          | ~400 tok/s    |
| 32B        | 32        | 18GB+       | 1          | Too tight     |

**Your GPU (16GB):** Can train up to 14B with rank 64, or 7B with rank 128.

## Implementation

### 1. Install Dependencies

```bash
pip install transformers peft accelerate bitsandbytes
pip install trl datasets evaluate  # Training utilities
```

### 2. Training Script

See `src/agents/training/lora_trainer.py` for full implementation.

Key components:
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 14,167,584,768 || trainable%: 0.03%
```

### 3. Data Format

Convert JSONL to HuggingFace format:
```python
def format_sample(sample):
    """Format instruction-tuning sample for training."""
    instruction = sample["instruction"]
    input_text = sample["input"]
    output = sample["output"]

    # Qwen2.5 chat format
    messages = [
        {"role": "system", "content": "You are an expert SNES developer."},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)
```

### 4. Training Command

```bash
cd ~/Code/hafs
python -m agents.training.lora_trainer \
  --base_model Qwen/Qwen2.5-Coder-14B-Instruct \
  --dataset ~/.context/training/pilot_hybrid_100_*/train.jsonl \
  --output_dir ~/.context/training/checkpoints/hafs-coder-v1 \
  --lora_r 64 \
  --lora_alpha 128 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_epochs 3 \
  --fp16 \
  --gradient_checkpointing
```

### 5. Monitor Training

Track loss curves:
```bash
tensorboard --logdir ~/.context/training/checkpoints/hafs-coder-v1/runs
```

## Merging Adapters

After training, merge LoRA weights into base model:

```python
from peft import PeftModel

# Load base + adapters
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
model = PeftModel.from_pretrained(base_model, "~/.context/training/checkpoints/hafs-coder-v1")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("~/.context/models/hafs-coder-14b-merged")
tokenizer.save_pretrained("~/.context/models/hafs-coder-14b-merged")
```

Or use adapters directly (faster inference):
```python
# Just load adapters at runtime
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
model = PeftModel.from_pretrained(model, "~/.context/training/checkpoints/hafs-coder-v1")
```

## Hyperparameter Tuning

### Learning Rate

Too high → divergence, too low → slow convergence
```python
# Try these in order:
learning_rates = [1e-4, 2e-4, 5e-4, 1e-3]
```

Monitor validation loss - should decrease steadily.

### LoRA Rank

Higher rank = more capacity, but risk overfitting on small datasets:
- **r=8-16:** Very small datasets (<1K samples)
- **r=32-64:** Medium datasets (1K-10K samples) ← Your case
- **r=128-256:** Large datasets (10K+ samples)

### Epochs

More epochs on small datasets → overfitting
```python
# Rule of thumb:
# - <1K samples: 5-10 epochs
# - 1K-10K samples: 3-5 epochs
# - 10K+ samples: 1-3 epochs
```

## Common Issues

### OOM (Out of Memory)

Solutions:
1. Reduce batch size: `--batch_size 2` or `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Enable gradient checkpointing: `--gradient_checkpointing`
4. Reduce LoRA rank: `--lora_r 32`
5. Use 8-bit quantization (slight quality loss):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       base_model,
       load_in_8bit=True,
       device_map="auto",
   )
   model = prepare_model_for_kbit_training(model)
   ```

### Loss Not Decreasing

- Learning rate too low → increase to 5e-4
- Dataset quality issues → check samples manually
- Not enough training → increase epochs

### Overfitting

Signs: train loss decreasing, val loss increasing
Solutions:
- Reduce epochs
- Add dropout: `lora_dropout=0.1`
- Add weight decay: `weight_decay=0.05`
- Get more data

## Serving Fine-Tuned Models

### Option 1: Ollama (Recommended)

Create Modelfile:
```dockerfile
FROM ~/.context/models/hafs-coder-14b-merged

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM """You are an expert SNES developer specializing in Zelda: A Link to the Past."""
```

Import to Ollama:
```bash
ollama create hafs-coder:14b -f Modelfile
ollama run hafs-coder:14b "Explain the NMI interrupt handler"
```

### Option 2: vLLM (High Performance)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model ~/.context/models/hafs-coder-14b-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16
```

### Option 3: HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "~/.context/models/hafs-coder-14b-merged",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("~/.context/models/hafs-coder-14b-merged")

# Generate
messages = [{"role": "user", "content": "Write a sprite DMA routine"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Next Steps

1. **Generate training data:** Use hybrid campaigns (in progress)
2. **Train LoRA adapter:** `python -m agents.training.lora_trainer`
3. **Evaluate model:** See `EVALUATION.md`
4. **Iterate:** Analyze failures, generate more targeted data, retrain
5. **Deploy:** Export to Ollama or serve with vLLM

## Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 Training Guide](https://qwen.readthedocs.io/en/latest/training/)
