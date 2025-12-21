"""Training checkpoint management for resilience and recovery."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""

    epoch: int
    step: int
    global_step: int
    loss: float
    learning_rate: float
    timestamp: float
    samples_seen: int
    model_name: str
    dataset_name: str
    best_loss: float
    early_stop_counter: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages training checkpoints for resilience and recovery.

    Features:
    - Automatic checkpointing every N steps
    - Keep last K checkpoints (configurable)
    - Save best checkpoint separately
    - Automatic recovery from latest checkpoint
    - Checkpoint validation

    Example:
        manager = CheckpointManager(
            checkpoint_dir=Path("~/.context/models/hyrule-asm-v1/checkpoints"),
            keep_last=5,
            save_best=True,
        )

        # During training
        manager.save_checkpoint(
            epoch=1,
            step=100,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            loss=2.5,
        )

        # Resume from checkpoint
        checkpoint = manager.load_latest_checkpoint()
        model.load_state_dict(checkpoint["model_state"])
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 5,
        save_best: bool = True,
        auto_clean: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            keep_last: Number of recent checkpoints to keep.
            save_best: Whether to save best checkpoint separately.
            auto_clean: Whether to auto-delete old checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.keep_last = keep_last
        self.save_best = save_best
        self.auto_clean = auto_clean

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")
        self.checkpoints: list[Path] = []

        logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        loss: float,
        learning_rate: float = 0.0,
        samples_seen: int = 0,
        model_name: str = "unknown",
        dataset_name: str = "unknown",
        early_stop_counter: int = 0,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch.
            step: Current step within epoch.
            model_state: Model state dict.
            optimizer_state: Optimizer state dict.
            loss: Current loss value.
            learning_rate: Current learning rate.
            samples_seen: Total samples processed.
            model_name: Name of model being trained.
            dataset_name: Name of dataset being used.
            early_stop_counter: Early stopping patience counter.
            additional_data: Additional data to save.

        Returns:
            Path to saved checkpoint.
        """
        # Create checkpoint name
        checkpoint_name = f"checkpoint_epoch{epoch:03d}_step{step:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Calculate global step
        global_step = samples_seen

        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            global_step=global_step,
            loss=loss,
            learning_rate=learning_rate,
            timestamp=time.time(),
            samples_seen=samples_seen,
            model_name=model_name,
            dataset_name=dataset_name,
            best_loss=self.best_loss,
            early_stop_counter=early_stop_counter,
        )

        # Prepare checkpoint data
        checkpoint_data = {
            "metadata": metadata.to_dict(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }

        if additional_data:
            checkpoint_data["additional_data"] = additional_data

        # Save checkpoint
        try:
            # Use torch if available, otherwise pickle
            try:
                import torch
                torch.save(checkpoint_data, checkpoint_path)
            except ImportError:
                import pickle
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

            logger.info(
                f"✓ Saved checkpoint: epoch={epoch}, step={step}, "
                f"loss={loss:.4f} → {checkpoint_path.name}"
            )

            # Track checkpoint
            self.checkpoints.append(checkpoint_path)

            # Save best checkpoint if this is the best loss
            if self.save_best and loss < self.best_loss:
                self.best_loss = loss
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"✓ New best checkpoint: loss={loss:.4f}")

            # Save metadata separately for quick inspection
            metadata_path = checkpoint_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Clean old checkpoints
            if self.auto_clean:
                self._clean_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"✗ Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Checkpoint data dict with keys:
            - metadata: CheckpointMetadata
            - model_state: Model state dict
            - optimizer_state: Optimizer state dict
            - additional_data: Any additional data (if saved)
        """
        try:
            # Load checkpoint
            try:
                import torch
                checkpoint_data = torch.load(checkpoint_path)
            except ImportError:
                import pickle
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)

            # Reconstruct metadata
            checkpoint_data["metadata"] = CheckpointMetadata.from_dict(
                checkpoint_data["metadata"]
            )

            logger.info(
                f"✓ Loaded checkpoint: {checkpoint_path.name} "
                f"(epoch={checkpoint_data['metadata'].epoch}, "
                f"step={checkpoint_data['metadata'].step})"
            )

            return checkpoint_data

        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def load_latest_checkpoint(self) -> Optional[dict[str, Any]]:
        """Load the most recent checkpoint.

        Returns:
            Checkpoint data or None if no checkpoints exist.
        """
        checkpoints = self._get_all_checkpoints()

        if not checkpoints:
            logger.warning("No checkpoints found")
            return None

        # Sort by modification time (newest first)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

        return self.load_checkpoint(latest)

    def load_best_checkpoint(self) -> Optional[dict[str, Any]]:
        """Load the best checkpoint (lowest loss).

        Returns:
            Checkpoint data or None if no best checkpoint exists.
        """
        best_path = self.checkpoint_dir / "checkpoint_best.pt"

        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None

        return self.load_checkpoint(best_path)

    def resume_training(
        self,
        model: Any,
        optimizer: Any,
        prefer_best: bool = False,
    ) -> Optional[CheckpointMetadata]:
        """Resume training from latest or best checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            prefer_best: If True, prefer best checkpoint over latest.

        Returns:
            Metadata from loaded checkpoint, or None if no checkpoint.
        """
        if prefer_best:
            checkpoint = self.load_best_checkpoint()
            if not checkpoint:
                checkpoint = self.load_latest_checkpoint()
        else:
            checkpoint = self.load_latest_checkpoint()

        if not checkpoint:
            logger.info("No checkpoint to resume from, starting fresh")
            return None

        # Load states
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        metadata = checkpoint["metadata"]

        logger.info(
            f"✓ Resumed training from epoch={metadata.epoch}, "
            f"step={metadata.step}, loss={metadata.loss:.4f}"
        )

        return metadata

    def _get_all_checkpoints(self) -> list[Path]:
        """Get all checkpoint files."""
        return sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))

    def _clean_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only last K."""
        checkpoints = self._get_all_checkpoints()

        if len(checkpoints) <= self.keep_last:
            return

        # Sort by modification time (oldest first)
        checkpoints_sorted = sorted(checkpoints, key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints
        to_remove = checkpoints_sorted[:-self.keep_last]

        for checkpoint_path in to_remove:
            # Also remove metadata JSON
            metadata_path = checkpoint_path.with_suffix(".json")

            checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            logger.debug(f"Removed old checkpoint: {checkpoint_path.name}")

        logger.info(f"Cleaned {len(to_remove)} old checkpoints, kept {self.keep_last}")

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints with metadata.

        Returns:
            List of checkpoint info dicts.
        """
        checkpoints = self._get_all_checkpoints()
        checkpoint_info = []

        for checkpoint_path in checkpoints:
            metadata_path = checkpoint_path.with_suffix(".json")

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                checkpoint_info.append({
                    "path": str(checkpoint_path),
                    "name": checkpoint_path.name,
                    "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                    "modified": checkpoint_path.stat().st_mtime,
                    **metadata,
                })

        return checkpoint_info

    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            True if checkpoint is valid, False otherwise.
        """
        try:
            checkpoint = self.load_checkpoint(checkpoint_path)

            # Check required keys
            required_keys = ["metadata", "model_state", "optimizer_state"]
            if not all(key in checkpoint for key in required_keys):
                logger.error(f"Checkpoint missing required keys: {checkpoint_path}")
                return False

            # Check metadata
            metadata = checkpoint["metadata"]
            if not isinstance(metadata, CheckpointMetadata):
                logger.error(f"Invalid metadata in checkpoint: {checkpoint_path}")
                return False

            logger.info(f"✓ Checkpoint validation passed: {checkpoint_path.name}")
            return True

        except Exception as e:
            logger.error(f"✗ Checkpoint validation failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir=Path("~/.context/models/test/checkpoints"),
        keep_last=3,
        save_best=True,
    )

    # Simulate training
    for epoch in range(3):
        for step in range(5):
            loss = 10.0 - (epoch * 5 + step) * 0.1  # Decreasing loss

            # Save checkpoint
            manager.save_checkpoint(
                epoch=epoch,
                step=step,
                model_state={"dummy": "model_state"},
                optimizer_state={"dummy": "optimizer_state"},
                loss=loss,
                learning_rate=0.001,
                samples_seen=epoch * 5 + step,
                model_name="test-model",
                dataset_name="test-dataset",
            )

    # List checkpoints
    print("\nCheckpoints:")
    for info in manager.list_checkpoints():
        print(f"  {info['name']}: loss={info['loss']:.4f}")

    # Load latest
    latest = manager.load_latest_checkpoint()
    if latest:
        print(f"\nLatest checkpoint: epoch={latest['metadata'].epoch}")

    # Load best
    best = manager.load_best_checkpoint()
    if best:
        print(f"Best checkpoint: loss={best['metadata'].loss:.4f}")
