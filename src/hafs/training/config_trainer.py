"""Configuration-based training system for Oracle experts.

Reads training.toml and executes training with appropriate hardware/model settings.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware configuration profile."""

    name: str
    device: str
    available_memory_gb: int
    max_batch_size: int
    max_sequence_length: int
    supports_fp16: bool
    supports_bf16: bool
    supports_gradient_checkpointing: bool
    max_lora_rank: int
    recommended_gradient_accumulation: int
    status: str = "available"
    reason: str = ""
    cost_per_hour: float = 0.0


@dataclass
class ExpertConfig:
    """Oracle expert configuration."""

    display_name: str
    role: str
    group: str
    base_model: str
    context_window: int
    specialization: str
    prompt_template: str = ""


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    expert: ExpertConfig
    hardware: HardwareProfile
    dataset_path: Path
    output_dir: Path

    # LoRA config
    lora_r: int
    lora_alpha: int
    lora_target_modules: list[str]
    lora_dropout: float
    lora_bias: str

    # Training hyperparameters
    num_epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    sequence_length: int
    warmup_steps: int
    weight_decay: float
    lr_scheduler: str
    logging_steps: int
    save_steps: int
    save_total_limit: int

    # Hardware-specific
    use_fp16: bool
    use_bf16: bool
    use_gradient_checkpointing: bool


class ConfigTrainer:
    """Configuration-based trainer for Oracle experts."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize trainer with config file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "training.toml"

        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load training configuration from TOML."""
        with open(self.config_path, "rb") as f:
            return tomllib.load(f)

    def get_hardware_profile(self, name: str) -> HardwareProfile:
        """Get hardware profile by name."""
        hw_data = self.config_data["hardware"][name]
        return HardwareProfile(
            name=hw_data["name"],
            device=hw_data["device"],
            available_memory_gb=hw_data["available_memory_gb"],
            max_batch_size=hw_data["max_batch_size"],
            max_sequence_length=hw_data["max_sequence_length"],
            supports_fp16=hw_data["supports_fp16"],
            supports_bf16=hw_data["supports_bf16"],
            supports_gradient_checkpointing=hw_data["supports_gradient_checkpointing"],
            max_lora_rank=hw_data["max_lora_rank"],
            recommended_gradient_accumulation=hw_data["recommended_gradient_accumulation"],
            status=hw_data.get("status", "available"),
            reason=hw_data.get("reason", ""),
            cost_per_hour=hw_data.get("cost_per_hour", 0.0),
        )

    def get_expert_config(self, name: str) -> ExpertConfig:
        """Get expert configuration by name."""
        expert_data = self.config_data["experts"][name]
        return ExpertConfig(
            display_name=expert_data["display_name"],
            role=expert_data["role"],
            group=expert_data["group"],
            base_model=expert_data["base_model"],
            context_window=expert_data["context_window"],
            specialization=expert_data["specialization"],
            prompt_template=expert_data.get("prompt_template", ""),
        )

    def build_training_config(self, preset_name: str) -> TrainingConfig:
        """Build complete training config from preset."""
        preset = self.config_data["presets"][preset_name]

        # Load components
        expert = self.get_expert_config(preset["expert"])
        hardware = self.get_hardware_profile(preset["hardware"])

        # Get hyperparameters
        hyper_name = preset["hyperparameters"]
        hyper = self.config_data["hyperparameters"][hyper_name]

        # Get LoRA config
        lora_name = preset["lora"]
        lora = self.config_data["lora"][lora_name]

        # Get dataset
        dataset_name = preset["dataset"]
        dataset_data = self.config_data["datasets"][dataset_name]
        dataset_path = Path(dataset_data["path"]).expanduser()

        # Build output directory
        date = datetime.now().strftime(self.config_data["output"]["date_format"])
        base_short = expert.base_model.split("/")[-1].lower().replace(".", "")
        expert_name = preset["expert"].replace("oracle-", "").split("-")[0]  # e.g., "rauru"

        model_name = f"oracle-{expert_name}-{expert.role}-{base_short}-{date}"
        output_dir = Path(self.config_data["output"]["models_dir"]).expanduser() / model_name

        return TrainingConfig(
            expert=expert,
            hardware=hardware,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            lora_target_modules=lora["target_modules"],
            lora_dropout=lora["lora_dropout"],
            lora_bias=lora["bias"],
            num_epochs=hyper["num_epochs"],
            learning_rate=hyper["learning_rate"],
            batch_size=preset["batch_size"],
            gradient_accumulation=preset["gradient_accumulation"],
            sequence_length=preset["sequence_length"],
            warmup_steps=hyper["warmup_steps"],
            weight_decay=hyper["weight_decay"],
            lr_scheduler=hyper["lr_scheduler"],
            logging_steps=hyper["logging_steps"],
            save_steps=hyper["save_steps"],
            save_total_limit=hyper["save_total_limit"],
            # Prefer bf16 over fp16 if both supported (better training stability)
            use_fp16=(hardware.supports_fp16 and not hardware.supports_bf16 and hardware.device != "cpu"),
            use_bf16=(hardware.supports_bf16 and hardware.device != "cpu"),
            use_gradient_checkpointing=hardware.supports_gradient_checkpointing,
        )

    def train(self, preset_name: str) -> Path:
        """Execute training with given preset."""
        config = self.build_training_config(preset_name)

        # Verify hardware
        if config.hardware.status != "available":
            raise RuntimeError(
                f"Hardware {config.hardware.name} is {config.hardware.status}: "
                f"{config.hardware.reason}"
            )

        # Print configuration
        self._print_config(config)

        # Check device availability
        device = self._verify_device(config.hardware.device)

        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.expert.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.expert.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info(f"Loading model: {config.expert.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            config.expert.base_model,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
            device_map=None if device == "mps" else "auto",
            trust_remote_code=True,
        )

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        logger.info(f"Loading dataset: {config.dataset_path}")
        dataset = load_dataset(
            "json",
            data_files={"train": str(config.dataset_path / "train.jsonl")},
        )["train"]

        # Tokenize dataset
        def format_sample(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            # Use expert's prompt template if available
            if config.expert.prompt_template:
                context = ", paired with context" if input_text else ""
                input_section = f"\n### Input:\n{input_text}\n" if input_text else ""
                prompt = config.expert.prompt_template.format(
                    context=context,
                    instruction=instruction,
                    input_section=input_section,
                    output=output,
                )
            else:
                # Default template
                if input_text:
                    prompt = f"""Below is an instruction, paired with context. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
                else:
                    prompt = f"""Below is an instruction. Write a response.

### Instruction:
{instruction}

### Response:
{output}"""

            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=config.sequence_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            format_sample,
            remove_columns=dataset.column_names,
            batched=False,
        )

        # Training arguments
        config.output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler,
            fp16=config.use_fp16,
            bf16=config.use_bf16,
            logging_steps=config.logging_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=config.use_gradient_checkpointing,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train
        logger.info("Starting training...")
        start_time = datetime.now()
        trainer.train()
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Training completed in {duration:.1f}s ({duration/60:.1f} minutes)")

        # Save model
        model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))

        # Save metadata
        self._save_metadata(config, duration, len(dataset))

        return config.output_dir

    def _verify_device(self, device: str) -> str:
        """Verify device availability."""
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            logger.info("Using MPS (Metal)")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"

    def _print_config(self, config: TrainingConfig):
        """Print training configuration."""
        print("=" * 80)
        print(f"Training Oracle Expert: {config.expert.display_name}")
        print("=" * 80)
        print(f"Role:         {config.expert.role}")
        print(f"Group:        {config.expert.group}")
        print(f"Base Model:   {config.expert.base_model}")
        print(f"Hardware:     {config.hardware.name}")
        print(f"Device:       {config.hardware.device}")
        print(f"Dataset:      {config.dataset_path}")
        print(f"Output:       {config.output_dir}")
        print()
        print("LoRA Configuration:")
        print(f"  Rank:       {config.lora_r}")
        print(f"  Alpha:      {config.lora_alpha}")
        print(f"  Targets:    {', '.join(config.lora_target_modules)}")
        print()
        print("Training Hyperparameters:")
        print(f"  Epochs:     {config.num_epochs}")
        print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation})")
        print(f"  Seq length: {config.sequence_length}")
        print(f"  Learn rate: {config.learning_rate}")
        print(f"  FP16:       {config.use_fp16}")
        print(f"  BF16:       {config.use_bf16}")
        print()

    def _save_metadata(self, config: TrainingConfig, duration: float, num_samples: int):
        """Save training metadata."""
        # Get git commit if available
        git_commit = "unknown"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:8]
        except Exception:
            pass

        metadata = {
            "expert": {
                "name": config.expert.display_name,
                "role": config.expert.role,
                "group": config.expert.group,
                "specialization": config.expert.specialization,
            },
            "model": {
                "base": config.expert.base_model,
                "lora_rank": config.lora_r,
                "lora_alpha": config.lora_alpha,
            },
            "training": {
                "dataset": str(config.dataset_path),
                "num_samples": num_samples,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "gradient_accumulation": config.gradient_accumulation,
                "learning_rate": config.learning_rate,
                "sequence_length": config.sequence_length,
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
            },
            "hardware": {
                "name": config.hardware.name,
                "device": config.hardware.device,
                "fp16": config.use_fp16,
                "bf16": config.use_bf16,
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "git_commit": git_commit,
            },
        }

        import json
        metadata_file = config.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_file}")
