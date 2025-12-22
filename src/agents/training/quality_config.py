"""Quality pipeline configuration loader.

Loads thresholds from training.toml and provides logging for tracking.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_training_config_path(
    config_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the training.toml path from args, env, or repo defaults."""
    if config_path is not None:
        return Path(config_path).expanduser()

    env_path = os.environ.get("HAFS_TRAINING_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    try:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / "config" / "training.toml"
        if candidate.exists():
            return candidate
    except Exception:
        candidate = None

    cwd_candidate = Path.cwd() / "config" / "training.toml"
    if cwd_candidate.exists():
        return cwd_candidate

    return candidate


@dataclass
class QualityConfig:
    """Configuration for quality pipeline thresholds."""

    # Quality thresholds
    quality_thresholds: dict[str, float] = field(default_factory=dict)

    # Dedup similarity thresholds
    dedup_thresholds: dict[str, float] = field(default_factory=dict)

    # KG validation settings
    kg_coverage: dict[str, float] = field(default_factory=dict)

    # Hallucination settings
    max_hallucination_risk: float = 0.5
    enable_multi_model_validation: bool = False

    # Metadata
    config_version: str = ""
    loaded_from: str = ""
    loaded_at: str = ""

    @classmethod
    def from_toml(cls, config_path: Optional[Path] = None) -> "QualityConfig":
        """Load configuration from training.toml."""
        config_path = resolve_training_config_path(config_path)
        if config_path is None or not config_path.exists():
            logger.warning("Training config not found, using defaults")
            return cls.defaults()

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls.defaults()

        generation = data.get("generation", {})

        # Load quality thresholds
        quality = generation.get("quality", {})
        quality_thresholds = {
            "default": quality.get("default_threshold", 0.4),
            "asm": quality.get("asm_threshold", 0.4),
            "oracle": quality.get("oracle_threshold", 0.4),
            "gigaleak": quality.get("gigaleak_threshold", 0.45),
            "cross_domain": quality.get("cross_domain_threshold", 0.5),
        }

        # Load dedup thresholds
        dedup = generation.get("dedup", {})
        dedup_thresholds = {
            "default": dedup.get("default_similarity", 0.88),
            "asm": dedup.get("asm_similarity", 0.85),
            "oracle": dedup.get("oracle_similarity", 0.85),
            "gigaleak": dedup.get("gigaleak_similarity", 0.88),
            "text": dedup.get("text_similarity", 0.82),
            "cpp": dedup.get("cpp_similarity", 0.88),
            "errors": dedup.get("errors_similarity", 0.80),
        }

        # Load KG coverage
        kg = generation.get("kg_validation", {})
        kg_coverage = {
            "default": kg.get("min_entity_coverage", 0.5),
            "asm": kg.get("asm_entity_coverage", 0.5),
            "oracle": kg.get("oracle_entity_coverage", 0.5),
            "gigaleak": kg.get("gigaleak_entity_coverage", 0.4),
            "text": kg.get("text_entity_coverage", 0.6),
        }

        # Load hallucination settings
        halluc = generation.get("hallucination", {})

        config = cls(
            quality_thresholds=quality_thresholds,
            dedup_thresholds=dedup_thresholds,
            kg_coverage=kg_coverage,
            max_hallucination_risk=halluc.get("max_risk", 0.5),
            enable_multi_model_validation=halluc.get("enable_multi_model", False),
            config_version=data.get("metadata", {}).get("schema_version", "unknown"),
            loaded_from=str(config_path),
            loaded_at=datetime.now().isoformat(),
        )

        logger.info(f"Loaded quality config from {config_path}")
        return config

    @classmethod
    def defaults(cls) -> "QualityConfig":
        """Return default configuration."""
        return cls(
            quality_thresholds={"default": 0.4, "asm": 0.4, "oracle": 0.4},
            dedup_thresholds={"default": 0.88, "asm": 0.85, "oracle": 0.85},
            kg_coverage={"default": 0.5},
            max_hallucination_risk=0.5,
            enable_multi_model_validation=False,
            config_version="defaults",
            loaded_from="",
            loaded_at=datetime.now().isoformat(),
        )

    def get_quality_threshold(self, domain: str) -> float:
        """Get quality threshold for a domain."""
        return self.quality_thresholds.get(domain, self.quality_thresholds["default"])

    def get_dedup_threshold(self, domain: str) -> float:
        """Get deduplication threshold for a domain."""
        return self.dedup_thresholds.get(domain, self.dedup_thresholds["default"])

    def get_kg_coverage(self, domain: str) -> float:
        """Get KG coverage requirement for a domain."""
        return self.kg_coverage.get(domain, self.kg_coverage["default"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for logging."""
        return {
            "quality_thresholds": self.quality_thresholds,
            "dedup_thresholds": self.dedup_thresholds,
            "kg_coverage": self.kg_coverage,
            "max_hallucination_risk": self.max_hallucination_risk,
            "enable_multi_model_validation": self.enable_multi_model_validation,
            "config_version": self.config_version,
            "loaded_from": self.loaded_from,
            "loaded_at": self.loaded_at,
        }

    def log_config(self) -> None:
        """Log configuration for tracking."""
        logger.info("=" * 60)
        logger.info("QUALITY PIPELINE CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Config version: {self.config_version}")
        logger.info(f"Loaded from: {self.loaded_from}")
        logger.info(f"Loaded at: {self.loaded_at}")
        logger.info("")
        logger.info("Quality Thresholds:")
        for domain, thresh in self.quality_thresholds.items():
            logger.info(f"  {domain}: {thresh}")
        logger.info("")
        logger.info("Dedup Thresholds (similarity):")
        for domain, thresh in self.dedup_thresholds.items():
            logger.info(f"  {domain}: {thresh}")
        logger.info("")
        logger.info("KG Coverage Requirements:")
        for domain, cov in self.kg_coverage.items():
            logger.info(f"  {domain}: {cov}")
        logger.info("")
        logger.info(f"Max Hallucination Risk: {self.max_hallucination_risk}")
        logger.info(f"Multi-Model Validation: {self.enable_multi_model_validation}")
        logger.info("=" * 60)


def save_generation_report(
    config: QualityConfig,
    generation_stats: dict[str, Any],
    output_path: Path,
) -> None:
    """Save generation report with config and stats for evaluation.

    Args:
        config: Quality configuration used
        generation_stats: Stats from generation run
        output_path: Where to save the report
    """
    import json

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "stats": generation_stats,
    }

    report_file = output_path / "generation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved generation report to {report_file}")
