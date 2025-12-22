"""Base classes for training data generation.

Provides abstract DataGenerator class and data models for training samples.
All domain-specific generators inherit from DataGenerator.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, TYPE_CHECKING

import uuid

if TYPE_CHECKING:
    from agents.training.provider_rotation import ProviderRotation

from agents.autonomy.base import MemoryAwareAgent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SourceItem")


@dataclass
class SourceItem:
    """Base class for domain-specific source items."""

    name: str
    content: str
    source: str  # Origin (e.g., "vanilla", "hack", file path)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        """Generate unique ID for this item."""
        return f"{self.source}:{self.name}"


@dataclass
class QualityScore:
    """Quality assessment for a training sample."""

    diversity_score: float = 0.0  # Embedding distance from existing samples
    kg_consistency: float = 0.0  # Knowledge graph validation score
    hallucination_risk: float = 0.0  # Risk of hallucinated content (lower is better)
    semantic_coherence: float = 0.0  # Input/output alignment

    @property
    def overall(self) -> float:
        """Compute weighted overall score."""
        return (
            0.30 * self.diversity_score
            + 0.20 * self.kg_consistency
            + 0.30 * (1.0 - self.hallucination_risk)
            + 0.20 * self.semantic_coherence
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "diversity_score": self.diversity_score,
            "kg_consistency": self.kg_consistency,
            "hallucination_risk": self.hallucination_risk,
            "semantic_coherence": self.semantic_coherence,
            "overall": self.overall,
        }


@dataclass
class TrainingSample:
    """Universal training sample format compatible with Unsloth.

    Follows the Alpaca instruction format by default but supports
    conversion to other formats (ChatML, Llama3, Qwen).
    """

    instruction: str  # User prompt / task description
    input: str  # Optional context or input data
    output: str  # Expected model response

    # Metadata
    domain: str  # "asm", "cpp", "text"
    source: str  # KB or file source
    sample_id: str = ""  # Unique identifier (ULID)

    # Quality metrics
    quality_score: float = 0.0
    embedding: Optional[list[float]] = None

    # Provenance
    teacher_model: str = ""
    teacher_prompt: str = ""
    timestamp: str = ""

    # Knowledge graph
    kg_entities: list[str] = field(default_factory=list)
    kg_validated: bool = False

    def __post_init__(self):
        if not self.sample_id:
            # Use UUID4 instead of ULID to avoid MemoryView issues
            self.sample_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_alpaca(self) -> dict[str, str]:
        """Convert to Alpaca instruction format."""
        result = {
            "instruction": self.instruction,
            "output": self.output,
        }
        if self.input:
            result["input"] = self.input
        return result

    def to_chatml(self) -> dict[str, Any]:
        """Convert to ChatML conversation format."""
        messages = []

        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})

        return {"messages": messages}

    def to_unsloth_format(self, template: str = "alpaca") -> dict[str, Any]:
        """Convert to Unsloth-compatible format.

        Args:
            template: Format template ("alpaca", "chatml", "llama3", "qwen")

        Returns:
            Dictionary in the specified format
        """
        if template == "chatml":
            return self.to_chatml()
        elif template in ("llama3", "qwen"):
            # Llama3 and Qwen use ChatML-style format
            return self.to_chatml()
        else:
            # Default to Alpaca
            return self.to_alpaca()

    def to_jsonl_entry(self, template: str = "alpaca") -> str:
        """Convert to JSONL line for export."""
        data = self.to_unsloth_format(template)
        # Add metadata for tracking
        data["_metadata"] = {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "source": self.source,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
        }
        return json.dumps(data)

    def to_dict(self) -> dict[str, Any]:
        """Full serialization including all fields."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "domain": self.domain,
            "source": self.source,
            "sample_id": self.sample_id,
            "quality_score": self.quality_score,
            "embedding": self.embedding,
            "teacher_model": self.teacher_model,
            "teacher_prompt": self.teacher_prompt,
            "timestamp": self.timestamp,
            "kg_entities": self.kg_entities,
            "kg_validated": self.kg_validated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSample":
        """Deserialize from dictionary."""
        return cls(
            instruction=data.get("instruction", ""),
            input=data.get("input", ""),
            output=data.get("output", ""),
            domain=data.get("domain", ""),
            source=data.get("source", ""),
            sample_id=data.get("sample_id", ""),
            quality_score=data.get("quality_score", 0.0),
            embedding=data.get("embedding"),
            teacher_model=data.get("teacher_model", ""),
            teacher_prompt=data.get("teacher_prompt", ""),
            timestamp=data.get("timestamp", ""),
            kg_entities=data.get("kg_entities", []),
            kg_validated=data.get("kg_validated", False),
        )


@dataclass
class GenerationResult:
    """Result from a generation run."""

    samples: list[TrainingSample]
    processed: int
    skipped: int
    errors: int
    duration_seconds: float

    @property
    def success_rate(self) -> float:
        total = self.processed + self.skipped + self.errors
        return self.processed / total if total > 0 else 0.0


@dataclass
class GenerationCheckpoint:
    """Checkpoint for resumable generation."""

    domain: str
    processed_ids: set[str]
    last_item_id: str
    total_processed: int
    total_errors: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "processed_ids": list(self.processed_ids),
            "last_item_id": self.last_item_id,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationCheckpoint":
        return cls(
            domain=data.get("domain", ""),
            processed_ids=set(data.get("processed_ids", [])),
            last_item_id=data.get("last_item_id", ""),
            total_processed=data.get("total_processed", 0),
            total_errors=data.get("total_errors", 0),
            timestamp=data.get("timestamp", ""),
        )


class DataGenerator(MemoryAwareAgent, ABC):
    """Abstract base class for domain-specific training data generators.

    Provides common infrastructure for:
    - Extracting source items from domain-specific sources
    - Generating training samples via teacher LLM
    - Checkpointing and resume support
    - Progress tracking and metrics
    - Multi-model provider rotation (NEW)
    """

    def __init__(
        self,
        name: str,
        domain: str,
        teacher_tier: str = "coding",
        provider_rotation: Optional["ProviderRotation"] = None,
    ):
        """Initialize the data generator.

        Args:
            name: Agent name
            domain: Domain identifier (e.g., "asm", "cpp", "text")
            teacher_tier: Model tier for teacher LLM ("fast", "coding", "reasoning")
            provider_rotation: Optional provider rotation config for multi-model generation
        """
        super().__init__(name, f"Generate training data for {domain} domain")
        self.domain = domain
        self.teacher_tier = teacher_tier

        # Multi-model support
        if provider_rotation is None:
            try:
                from agents.training.provider_rotation import load_provider_rotation
                provider_rotation = load_provider_rotation()
            except Exception as e:
                logger.debug(f"Provider rotation config not loaded: {e}")
        self._provider_rotation = provider_rotation
        self._orchestrator = None

        # Paths
        self.training_dir = self.context_root / "training" / domain
        self.checkpoint_dir = self.training_dir / "checkpoints"
        self.output_dir = self.training_dir / "output"

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._checkpoint: Optional[GenerationCheckpoint] = None

    def set_provider_rotation(self, rotation: "ProviderRotation") -> None:
        """Set provider rotation for multi-model generation."""
        self._provider_rotation = rotation
        logger.info(f"Set provider rotation with {len(rotation.providers)} providers")

    async def generate_with_rotation(
        self,
        prompt: str,
        tier: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        """Generate response using provider rotation.

        Args:
            prompt: The prompt to send to the model
            tier: Override tier (defaults to self.teacher_tier)

        Returns:
            Tuple of (response_content, model_name) or (None, "") on failure
        """
        from agents.training.provider_rotation import get_provider_enum, get_tier_enum

        if not self._orchestrator:
            from core.orchestrator_v2 import UnifiedOrchestrator
            self._orchestrator = UnifiedOrchestrator()

        use_tier = tier or self.teacher_tier

        # If no rotation configured, use default Gemini
        if not self._provider_rotation:
            from core.orchestrator_v2 import Provider, TaskTier
            try:
                response = await self._orchestrator.generate(
                    prompt=prompt,
                    tier=get_tier_enum(use_tier),
                    provider=Provider.GEMINI,
                )
                return response.content, response.model
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return None, ""

        # Use rotation
        provider_config = self._provider_rotation.select_provider()
        if not provider_config:
            logger.error("No providers available in rotation")
            return None, ""

        try:
            provider_enum = get_provider_enum(provider_config.name)
            response = await self._orchestrator.generate(
                prompt=prompt,
                tier=get_tier_enum(use_tier),
                provider=provider_enum,
                model=provider_config.model or None,
            )

            model_name = response.model or provider_config.model or f"{provider_config.name}-default"
            return response.content, model_name

        except Exception as e:
            logger.warning(f"Provider {provider_config.name} failed: {e}")

            # Try fallback
            fallback = self._provider_rotation.get_fallback(provider_config.name)
            if fallback:
                try:
                    fallback_enum = get_provider_enum(fallback.name)
                    response = await self._orchestrator.generate(
                        prompt=prompt,
                        tier=get_tier_enum(use_tier),
                        provider=fallback_enum,
                        model=fallback.model or None,
                    )
                    model_name = response.model or fallback.model or f"{fallback.name}-default"
                    return response.content, model_name
                except Exception as e2:
                    logger.error(f"Fallback provider {fallback.name} also failed: {e2}")

            return None, ""

    @abstractmethod
    async def extract_source_items(self) -> list[SourceItem]:
        """Extract items from domain-specific sources.

        Returns:
            List of source items to process
        """
        pass

    @abstractmethod
    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Generate a training sample from a source item.

        Uses teacher LLM to synthesize instruction/input/output triplet.

        Args:
            item: Source item to process

        Returns:
            TrainingSample if successful, None if failed
        """
        pass

    @abstractmethod
    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Get the teacher LLM prompt for this item.

        Args:
            item: Source item to generate prompt for

        Returns:
            Prompt string for teacher model
        """
        pass

    def get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.checkpoint_dir / f"checkpoint_{self.domain}.json"

    def load_checkpoint(self) -> Optional[GenerationCheckpoint]:
        """Load checkpoint from disk if exists."""
        path = self.get_checkpoint_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._checkpoint = GenerationCheckpoint.from_dict(data)
                logger.info(
                    f"Loaded checkpoint: {self._checkpoint.total_processed} processed"
                )
                return self._checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def save_checkpoint(self, checkpoint: GenerationCheckpoint) -> None:
        """Save checkpoint to disk."""
        path = self.get_checkpoint_path()
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
        self._checkpoint = checkpoint

    def clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        path = self.get_checkpoint_path()
        if path.exists():
            path.unlink()
        self._checkpoint = None

    async def generate_batch(
        self,
        items: list[SourceItem],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TrainingSample]:
        """Generate samples from a batch of items with checkpointing.

        Args:
            items: List of source items to process
            batch_size: Number of items to process before checkpointing
            progress_callback: Optional callback (processed, total)

        Returns:
            List of generated samples
        """
        samples: list[TrainingSample] = []

        # Load checkpoint to get processed_ids for incremental updates
        # Note: items are already filtered by run_generation(), don't filter again
        checkpoint = self.load_checkpoint()
        processed_ids = checkpoint.processed_ids if checkpoint else set()

        total = len(items)
        processed = 0
        errors = 0

        for i, item in enumerate(items):
            try:
                sample = await self.generate_sample(item)
                if sample:
                    samples.append(sample)
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error(f"Error generating sample for {item.name}: {e}")
                errors += 1

            processed_ids.add(item.item_id)

            # Checkpoint every batch_size items
            if (i + 1) % batch_size == 0:
                self.save_checkpoint(
                    GenerationCheckpoint(
                        domain=self.domain,
                        processed_ids=processed_ids,
                        last_item_id=item.item_id,
                        total_processed=processed,
                        total_errors=errors,
                    )
                )

            if progress_callback:
                progress_callback(i + 1, total)

        # Final checkpoint
        self.save_checkpoint(
            GenerationCheckpoint(
                domain=self.domain,
                processed_ids=processed_ids,
                last_item_id=items[-1].item_id if items else "",
                total_processed=processed,
                total_errors=errors,
            )
        )

        return samples

    async def run_generation(
        self,
        limit: Optional[int] = None,
        resume: bool = True,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Run full generation pipeline.

        Args:
            limit: Maximum number of samples to generate
            resume: Whether to resume from checkpoint
            output_path: Optional output path for JSONL

        Returns:
            GenerationResult with samples and metrics
        """
        import time

        start_time = time.time()

        # Setup if needed
        if not hasattr(self, "orchestrator") or self.orchestrator is None:
            await self.setup()

        # Extract source items
        items = await self.extract_source_items()
        if limit:
            items = items[:limit]

        logger.info(f"Extracted {len(items)} source items for {self.domain}")

        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_ids = checkpoint.processed_ids
                items = [i for i in items if i.item_id not in processed_ids]
                logger.info(f"Resuming: {len(items)} items remaining")

        # Generate samples
        samples = await self.generate_batch(
            items,
            progress_callback=lambda p, t: logger.info(f"Progress: {p}/{t}"),
        )

        duration = time.time() - start_time

        # Save output if path provided
        if output_path:
            self._save_output(samples, output_path)

        # Clear checkpoint on success
        if len(samples) > 0:
            self.clear_checkpoint()

        return GenerationResult(
            samples=samples,
            processed=len(samples),
            skipped=0,
            errors=len(items) - len(samples),
            duration_seconds=duration,
        )

    def _save_output(
        self, samples: list[TrainingSample], path: Path, template: str = "alpaca"
    ) -> None:
        """Save samples to JSONL file."""
        with open(path, "w") as f:
            for sample in samples:
                f.write(sample.to_jsonl_entry(template) + "\n")
        logger.info(f"Saved {len(samples)} samples to {path}")

    async def run_task(self, task: dict[str, Any]) -> str:
        """Run generation task (BaseAgent interface)."""
        output = task.get("output", self.output_dir / f"{self.domain}_train.jsonl")
        limit = task.get("limit", 100)
        resume = task.get("resume", True)

        result = await self.run_generation(
            limit=limit,
            resume=resume,
            output_path=Path(output),
        )

        return (
            f"Generated {result.processed} samples in {result.duration_seconds:.1f}s "
            f"({result.errors} errors)"
        )
