"""Base validator interface for training samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from agents.training.base import TrainingSample


@dataclass
class ValidationResult:
    """Result of validating a training sample."""

    valid: bool
    score: float  # 0.0-1.0, higher is better
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


class Validator(ABC):
    """Abstract base class for domain-specific validators."""

    def __init__(self, name: str, domain: str):
        """Initialize validator.

        Args:
            name: Validator name
            domain: Domain this validator handles (asm, cpp, text, etc.)
        """
        self.name = name
        self.domain = domain

    @abstractmethod
    async def validate(self, sample: TrainingSample) -> ValidationResult:
        """Validate a training sample.

        Args:
            sample: The training sample to validate

        Returns:
            ValidationResult with validity, score, and any errors/warnings
        """
        pass

    def can_validate(self, sample: TrainingSample) -> bool:
        """Check if this validator can handle the sample.

        Args:
            sample: The training sample

        Returns:
            True if this validator can validate the sample
        """
        return sample.domain == self.domain

    async def validate_batch(
        self, samples: list[TrainingSample]
    ) -> list[ValidationResult]:
        """Validate multiple samples.

        Args:
            samples: List of samples to validate

        Returns:
            List of ValidationResults
        """
        results = []
        for sample in samples:
            if self.can_validate(sample):
                result = await self.validate(sample)
            else:
                result = ValidationResult(
                    valid=True,
                    score=1.0,
                    warnings=[f"Validator {self.name} skipped: wrong domain"],
                )
            results.append(result)
        return results


class CompositeValidator(Validator):
    """Combines multiple validators for comprehensive validation."""

    def __init__(self, validators: list[Validator]):
        """Initialize with list of validators.

        Args:
            validators: List of validators to apply
        """
        super().__init__("CompositeValidator", "all")
        self.validators = validators

    def can_validate(self, sample: TrainingSample) -> bool:
        """Check if any validator can handle this sample."""
        return any(v.can_validate(sample) for v in self.validators)

    async def validate(self, sample: TrainingSample) -> ValidationResult:
        """Apply all applicable validators and combine results."""
        applicable = [v for v in self.validators if v.can_validate(sample)]

        if not applicable:
            return ValidationResult(
                valid=True,
                score=1.0,
                warnings=["No applicable validators"],
            )

        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_details: dict[str, Any] = {}
        scores: list[float] = []

        for validator in applicable:
            result = await validator.validate(sample)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_details[validator.name] = result.to_dict()
            scores.append(result.score)

        # Overall validity requires all validators to pass
        valid = len(all_errors) == 0

        # Average score across validators
        avg_score = sum(scores) / len(scores) if scores else 1.0

        return ValidationResult(
            valid=valid,
            score=avg_score,
            errors=all_errors,
            warnings=all_warnings,
            details=all_details,
        )
