"""Training Exporter for Unsloth and other frameworks.

Exports training data in formats compatible with:
- Unsloth (Alpaca, ChatML, Llama3, Qwen templates)
- HuggingFace datasets
- Custom formats
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import TrainingSample

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    template: str = "alpaca"  # alpaca, chatml, llama3, qwen
    include_metadata: bool = False
    max_length: Optional[int] = None  # Truncate samples exceeding this
    add_eos_token: bool = True
    add_bos_token: bool = False


@dataclass
class ModelConfig:
    """Configuration for a target model."""

    name: str
    template: str
    context_length: int
    special_tokens: dict[str, str] = field(default_factory=dict)

    def get_max_length(self) -> int:
        """Get maximum sequence length (leave room for generation)."""
        return int(self.context_length * 0.75)


class TrainingExporter:
    """Export training data for Unsloth and other frameworks."""

    # Unsloth-compatible chat templates
    TEMPLATES = {
        "alpaca": {
            "instruction_start": "### Instruction:\n",
            "input_start": "\n### Input:\n",
            "response_start": "\n### Response:\n",
            "instruction_end": "",
            "response_end": "",
        },
        "chatml": {
            "instruction_start": "<|im_start|>user\n",
            "input_start": "",  # Input concatenated with instruction
            "response_start": "<|im_end|>\n<|im_start|>assistant\n",
            "instruction_end": "",
            "response_end": "<|im_end|>",
        },
        "llama3": {
            "instruction_start": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
            "input_start": "",
            "response_start": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "instruction_end": "",
            "response_end": "<|eot_id|>",
        },
        "qwen": {
            "instruction_start": "<|im_start|>user\n",
            "input_start": "",
            "response_start": "<|im_end|>\n<|im_start|>assistant\n",
            "instruction_end": "",
            "response_end": "<|im_end|>",
        },
        "mistral": {
            "instruction_start": "[INST] ",
            "input_start": "",
            "response_start": " [/INST]",
            "instruction_end": "",
            "response_end": "</s>",
        },
    }

    # Target model configurations
    MODEL_CONFIGS = {
        "qwen2.5-coder-7b": ModelConfig(
            name="qwen2.5-coder-7b",
            template="qwen",
            context_length=32768,
        ),
        "qwen2.5-coder-14b": ModelConfig(
            name="qwen2.5-coder-14b",
            template="qwen",
            context_length=32768,
        ),
        "deepseek-coder-v2-lite": ModelConfig(
            name="deepseek-coder-v2-lite",
            template="chatml",
            context_length=16384,
        ),
        "codellama-7b": ModelConfig(
            name="codellama-7b",
            template="llama3",
            context_length=16384,
        ),
        "codellama-13b": ModelConfig(
            name="codellama-13b",
            template="llama3",
            context_length=16384,
        ),
        "mistral-7b": ModelConfig(
            name="mistral-7b",
            template="mistral",
            context_length=32768,
        ),
        "mixtral-8x7b": ModelConfig(
            name="mixtral-8x7b",
            template="mistral",
            context_length=32768,
        ),
    }

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def format_sample(
        self,
        sample: TrainingSample,
        template: Optional[str] = None,
    ) -> str:
        """Format a sample using the specified template.

        Args:
            sample: Training sample to format
            template: Template name (alpaca, chatml, etc.)

        Returns:
            Formatted text string
        """
        template = template or self.config.template
        tmpl = self.TEMPLATES.get(template, self.TEMPLATES["alpaca"])

        # Build user content
        user_content = sample.instruction
        if sample.input:
            if tmpl["input_start"]:
                user_content += tmpl["input_start"] + sample.input
            else:
                user_content += "\n\n" + sample.input

        # Build full text
        text = (
            tmpl["instruction_start"]
            + user_content
            + tmpl["instruction_end"]
            + tmpl["response_start"]
            + sample.output
            + tmpl["response_end"]
        )

        # Truncate if needed
        if self.config.max_length and len(text) > self.config.max_length:
            text = text[: self.config.max_length]

        return text

    def sample_to_dict(
        self,
        sample: TrainingSample,
        template: Optional[str] = None,
        include_text: bool = True,
    ) -> dict[str, Any]:
        """Convert sample to dictionary for JSON export.

        Args:
            sample: Training sample
            template: Template name
            include_text: Whether to include formatted text field

        Returns:
            Dictionary representation
        """
        template = template or self.config.template

        result: dict[str, Any] = {}

        # For ChatML-style templates, use messages format
        if template in ("chatml", "qwen", "llama3"):
            messages = []

            user_content = sample.instruction
            if sample.input:
                user_content += "\n\n" + sample.input

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": sample.output})

            result["messages"] = messages

        # For Alpaca format, use instruction/input/output
        else:
            result["instruction"] = sample.instruction
            if sample.input:
                result["input"] = sample.input
            result["output"] = sample.output

        # Add formatted text if requested
        if include_text:
            result["text"] = self.format_sample(sample, template)

        # Add metadata if configured
        if self.config.include_metadata:
            result["_metadata"] = {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "source": sample.source,
                "quality_score": sample.quality_score,
            }

        return result

    def export_jsonl(
        self,
        samples: list[TrainingSample],
        output_path: Path,
        template: Optional[str] = None,
    ) -> int:
        """Export samples to JSONL file.

        Args:
            samples: List of samples to export
            output_path: Output file path
            template: Template to use

        Returns:
            Number of samples exported
        """
        template = template or self.config.template

        with open(output_path, "w") as f:
            for sample in samples:
                entry = self.sample_to_dict(sample, template, include_text=True)
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Exported {len(samples)} samples to {output_path}")
        return len(samples)

    def export_for_model(
        self,
        samples: list[TrainingSample],
        model: str,
        output_dir: Path,
        splits: Optional[dict[str, list[TrainingSample]]] = None,
    ) -> dict[str, Path]:
        """Export with model-specific formatting.

        Args:
            samples: All samples (used if splits not provided)
            model: Model name (e.g., "qwen2.5-coder-7b")
            output_dir: Output directory
            splits: Optional pre-split data {"train": [...], "val": [...], "test": [...]}

        Returns:
            Dictionary of output paths
        """
        model_config = self.MODEL_CONFIGS.get(model)
        if not model_config:
            logger.warning(f"Unknown model {model}, using default config")
            model_config = ModelConfig(
                name=model,
                template="alpaca",
                context_length=16384,
            )

        # Update config for this model
        self.config.template = model_config.template
        self.config.max_length = model_config.get_max_length()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, Path] = {}

        if splits:
            # Export each split
            for split_name, split_samples in splits.items():
                path = output_dir / f"{split_name}.jsonl"
                self.export_jsonl(split_samples, path)
                outputs[split_name] = path
        else:
            # Export all as single file
            path = output_dir / "train.jsonl"
            self.export_jsonl(samples, path)
            outputs["train"] = path

        # Save model config
        config_path = output_dir / "config.json"
        config_data = {
            "model": model,
            "template": model_config.template,
            "context_length": model_config.context_length,
            "max_train_length": model_config.get_max_length(),
            "sample_count": len(samples),
        }
        config_path.write_text(json.dumps(config_data, indent=2))
        outputs["config"] = config_path

        # Generate Unsloth training script
        script_path = output_dir / "train_unsloth.py"
        self._generate_unsloth_script(script_path, model, model_config)
        outputs["script"] = script_path

        return outputs

    def _generate_unsloth_script(
        self,
        path: Path,
        model: str,
        config: ModelConfig,
    ) -> None:
        """Generate a sample Unsloth training script."""
        script = f'''"""Unsloth training script for {model}.

Generated by hafs DataCurator.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Model configuration
MODEL_NAME = "{model}"
MAX_SEQ_LENGTH = {config.context_length}
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

def main():
    # Load model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for 16GB VRAM
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    dataset = load_dataset("json", data_files={{
        "train": "train.jsonl",
        "validation": "val.jsonl",
    }})

    # Format function for {config.template} template
    def format_prompt(examples):
        texts = examples["text"]
        return {{"text": texts}}

    dataset = dataset.map(format_prompt, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained("./lora_model")
    tokenizer.save_pretrained("./lora_model")

    # Optionally merge and save full model
    # model.save_pretrained_merged("./merged_model", tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    main()
'''
        path.write_text(script)
        logger.info(f"Generated training script: {path}")

    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return self.MODEL_CONFIGS.get(model)

    def list_models(self) -> list[str]:
        """List all supported models."""
        return list(self.MODEL_CONFIGS.keys())

    def list_templates(self) -> list[str]:
        """List all available templates."""
        return list(self.TEMPLATES.keys())
