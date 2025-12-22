"""Model Registry System for Trained Models.

Tracks trained models across multiple machines and provides:
- Model metadata and metrics
- Cross-platform model deployment
- Integration with halext nodes, ollama, llama.cpp
- Model versioning and lineage tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

ModelLocation = Literal["mac", "windows", "cloud", "halext"]
ModelFormat = Literal["pytorch", "gguf", "onnx", "safetensors"]
ServingBackend = Literal["ollama", "llama.cpp", "vllm", "transformers", "halext-node"]


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    # Identity
    model_id: str  # Unique ID: oracle-farore-general-qwen25-coder-15b-20251222
    display_name: str  # Oracle: Farore Secrets
    version: str  # v1, v2, etc.

    # Training info
    base_model: str  # Qwen/Qwen2.5-Coder-1.5B
    role: str  # general, asm, debug, yaze
    group: str  # rom-tooling
    training_date: str  # ISO format
    training_duration_minutes: int

    # Dataset info
    dataset_name: str
    dataset_path: str
    train_samples: int
    val_samples: int
    test_samples: int
    dataset_quality: dict[str, float]  # acceptance_rate, avg_diversity, etc.

    # Training metrics
    final_loss: Optional[float]
    best_loss: Optional[float]
    eval_loss: Optional[float]
    perplexity: Optional[float]

    # Model configuration
    lora_config: dict[str, Any]  # rank, alpha, target_modules, etc.
    hyperparameters: dict[str, Any]  # lr, batch_size, etc.

    # Hardware
    hardware: str  # windows-rtx-5060, mac-mps, cloud-a100, etc.
    device: str  # cuda, mps, cpu

    # Files
    model_path: str  # Full path on original machine
    checkpoint_path: Optional[str]
    adapter_path: Optional[str]  # LoRA adapter path

    # Formats available
    formats: list[ModelFormat] = field(default_factory=lambda: ["pytorch"])

    # Locations
    locations: dict[ModelLocation, str] = field(default_factory=dict)
    primary_location: ModelLocation = "windows"

    # Serving
    deployed_backends: list[ServingBackend] = field(default_factory=list)
    ollama_model_name: Optional[str] = None
    halext_node_id: Optional[str] = None

    # Metadata
    git_commit: Optional[str] = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(**data)


@dataclass
class ModelTransferStatus:
    """Status of a model transfer operation."""

    model_id: str
    source: ModelLocation
    destination: ModelLocation
    status: Literal["pending", "in_progress", "completed", "failed"]
    progress_percent: float
    bytes_transferred: int
    total_bytes: int
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class ModelRegistry:
    """Central registry for tracking trained models across machines."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize model registry.

        Args:
            registry_path: Path to registry JSON file. Defaults to ~/.context/models/registry.json
        """
        if registry_path is None:
            registry_path = Path.home() / ".context" / "models" / "registry.json"

        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, ModelMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if not self.registry_path.exists():
            logger.info("No registry found, creating new registry")
            self._save_registry()
            return

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            self.models = {
                model_id: ModelMetadata.from_dict(model_data)
                for model_id, model_data in data.get("models", {}).items()
            }

            logger.info(f"Loaded {len(self.models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.models = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "models": {
                    model_id: model.to_dict() for model_id, model in self.models.items()
                },
            }

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved registry with {len(self.models)} models")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def register_model(
        self,
        model_id: str,
        display_name: str,
        base_model: str,
        role: str,
        model_path: str,
        location: ModelLocation,
        **kwargs,
    ) -> ModelMetadata:
        """Register a new trained model.

        Args:
            model_id: Unique model identifier
            display_name: Human-readable name
            base_model: Base model name (e.g., Qwen/Qwen2.5-Coder-1.5B)
            role: Model role (general, asm, debug, etc.)
            model_path: Full path to model on original machine
            location: Primary location where model is stored
            **kwargs: Additional metadata fields

        Returns:
            ModelMetadata object
        """
        if model_id in self.models:
            logger.warning(f"Model {model_id} already registered, updating metadata")

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            base_model=base_model,
            role=role,
            model_path=model_path,
            primary_location=location,
            locations={location: model_path},
            **kwargs,
        )

        self.models[model_id] = metadata
        self._save_registry()

        logger.info(f"Registered model: {model_id} at {location}:{model_path}")
        return metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def list_models(
        self,
        role: Optional[str] = None,
        location: Optional[ModelLocation] = None,
        backend: Optional[ServingBackend] = None,
    ) -> list[ModelMetadata]:
        """List models matching filters.

        Args:
            role: Filter by role (asm, debug, general, etc.)
            location: Filter by location (mac, windows, cloud, halext)
            backend: Filter by deployed backend (ollama, llama.cpp, etc.)

        Returns:
            List of matching model metadata
        """
        models = list(self.models.values())

        if role:
            models = [m for m in models if m.role == role]

        if location:
            models = [m for m in models if location in m.locations]

        if backend:
            models = [m for m in models if backend in m.deployed_backends]

        return models

    def update_location(
        self, model_id: str, location: ModelLocation, path: str
    ) -> None:
        """Update model location after transfer.

        Args:
            model_id: Model identifier
            location: New location
            path: Path to model at new location
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in registry")

        model.locations[location] = path
        model.updated_at = datetime.now().isoformat()

        self._save_registry()
        logger.info(f"Updated {model_id} location: {location} -> {path}")

    def add_backend(
        self,
        model_id: str,
        backend: ServingBackend,
        backend_model_name: Optional[str] = None,
    ) -> None:
        """Mark model as deployed to a serving backend.

        Args:
            model_id: Model identifier
            backend: Serving backend (ollama, llama.cpp, etc.)
            backend_model_name: Model name in the backend system
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in registry")

        if backend not in model.deployed_backends:
            model.deployed_backends.append(backend)

        if backend == "ollama" and backend_model_name:
            model.ollama_model_name = backend_model_name
        elif backend == "halext-node" and backend_model_name:
            model.halext_node_id = backend_model_name

        model.updated_at = datetime.now().isoformat()
        self._save_registry()

        logger.info(f"Deployed {model_id} to {backend}")

    def get_available_locations(self, model_id: str) -> list[tuple[ModelLocation, str]]:
        """Get all locations where model is available.

        Returns:
            List of (location, path) tuples
        """
        model = self.models.get(model_id)
        if not model:
            return []

        return list(model.locations.items())

    def find_best_location(
        self, model_id: str, preferred: Optional[ModelLocation] = None
    ) -> Optional[tuple[ModelLocation, str]]:
        """Find best location to pull model from.

        Args:
            model_id: Model identifier
            preferred: Preferred location if available

        Returns:
            (location, path) tuple or None if not available
        """
        model = self.models.get(model_id)
        if not model:
            return None

        # Check preferred location first
        if preferred and preferred in model.locations:
            return (preferred, model.locations[preferred])

        # Try primary location
        if model.primary_location in model.locations:
            return (model.primary_location, model.locations[model.primary_location])

        # Return any available location
        if model.locations:
            location = list(model.locations.keys())[0]
            return (location, model.locations[location])

        return None

    def delete_model(self, model_id: str) -> None:
        """Remove model from registry (does not delete files)."""
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            logger.info(f"Deleted {model_id} from registry")

    def export_catalog(self, output_path: Path) -> None:
        """Export model catalog to JSON file."""
        catalog = {
            "exported_at": datetime.now().isoformat(),
            "total_models": len(self.models),
            "models": [model.to_dict() for model in self.models.values()],
        }

        with open(output_path, "w") as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Exported catalog to {output_path}")


def register_training_run(
    model_path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    location: ModelLocation = "windows",
) -> ModelMetadata:
    """Helper to register a training run.

    Args:
        model_path: Path to trained model directory
        config: Training configuration
        metrics: Training metrics from trainer_state.json
        location: Location where model was trained

    Returns:
        ModelMetadata object
    """
    registry = ModelRegistry()

    # Extract info from path: oracle-farore-general-qwen25-coder-15b-20251222
    model_name = model_path.name
    parts = model_name.split("-")

    # Build model_id and display name
    model_id = model_name
    display_name = config.get("display_name", model_name)

    # Register model
    metadata = registry.register_model(
        model_id=model_id,
        display_name=display_name,
        version="v1",
        base_model=config.get("base_model", "unknown"),
        role=config.get("role", "unknown"),
        group=config.get("group", "unknown"),
        training_date=datetime.now().isoformat(),
        training_duration_minutes=metrics.get("duration_minutes", 0),
        dataset_name=config.get("dataset", "unknown"),
        dataset_path=config.get("dataset_path", ""),
        train_samples=metrics.get("train_samples", 0),
        val_samples=metrics.get("val_samples", 0),
        test_samples=metrics.get("test_samples", 0),
        dataset_quality=metrics.get("quality", {}),
        final_loss=metrics.get("loss"),
        best_loss=metrics.get("best_loss"),
        lora_config=config.get("lora", {}),
        hyperparameters=config.get("hyperparameters", {}),
        hardware=config.get("hardware", "unknown"),
        device=config.get("device", "unknown"),
        model_path=str(model_path),
        location=location,
    )

    return metadata
