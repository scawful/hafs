"""Schema validation for cognitive protocol JSON files.

Provides runtime validation against Pydantic schemas with auto-fix capability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from core.protocol.io_manager import get_io_manager
from models.goals import GoalHierarchy
from models.metacognition import MetacognitiveState

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SchemaValidator:
    """Validates JSON files against Pydantic schemas.

    Provides:
    - Schema validation with helpful error messages
    - Auto-fix using default values for corrupted files
    - Directory-wide validation

    Example:
        validator = SchemaValidator()
        state = validator.validate_file(
            Path(".context/scratchpad/metacognition.json"),
            MetacognitiveState
        )
    """

    # Map filenames to their Pydantic schemas
    SCHEMAS: dict[str, Type[BaseModel]] = {
        "metacognition.json": MetacognitiveState,
        "goals.json": GoalHierarchy,
        # Add more as needed:
        # "emotions.json": EmotionalState,
        # "epistemic.json": EpistemicState,
    }

    @classmethod
    def validate_file(cls, file_path: Path, schema: Type[T]) -> T:
        """Validate JSON file against Pydantic schema.

        Args:
            file_path: Path to JSON file.
            schema: Pydantic model class to validate against.

        Returns:
            Validated model instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValidationError: If validation fails and can't auto-fix.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            io_manager = get_io_manager()
            data = io_manager.read_json(file_path)
            return schema.model_validate(data)

        except ValidationError as e:
            logger.error(f"Validation error in {file_path}: {e}")
            raise

    @classmethod
    def validate_file_with_auto_fix(
        cls, file_path: Path, schema: Type[T]
    ) -> tuple[T, bool]:
        """Validate file with auto-fix fallback.

        If validation fails, creates a new instance with defaults
        and optionally overwrites the file.

        Args:
            file_path: Path to JSON file.
            schema: Pydantic model class.

        Returns:
            Tuple of (model_instance, was_fixed).
            - model_instance: Valid model (either loaded or default)
            - was_fixed: True if file was invalid and defaults were used

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        try:
            model = cls.validate_file(file_path, schema)
            return model, False
        except ValidationError as e:
            logger.warning(
                f"Failed to validate {file_path}, using defaults: {e}"
            )
            # Create default instance
            model = schema()
            return model, True

    @classmethod
    def auto_fix_file(
        cls, file_path: Path, schema: Type[T], overwrite: bool = True
    ) -> bool:
        """Attempt to auto-fix a corrupted file.

        Creates a new instance with default values and optionally
        overwrites the file.

        Args:
            file_path: Path to file to fix.
            schema: Pydantic model class.
            overwrite: If True, overwrite the file with defaults.

        Returns:
            True if file was fixed successfully.
        """
        try:
            # Create default instance
            model = schema()

            if overwrite:
                io_manager = get_io_manager()
                data = model.model_dump(mode="json")
                io_manager.write_json(file_path, data, immediate=True)
                logger.info(f"Auto-fixed {file_path} with default values")

            return True

        except Exception as e:
            logger.error(f"Failed to auto-fix {file_path}: {e}")
            return False

    @classmethod
    def validate_directory(
        cls, context_dir: Path, auto_fix: bool = False
    ) -> dict[str, bool]:
        """Validate all known schema files in a directory.

        Args:
            context_dir: Path to .context directory.
            auto_fix: If True, auto-fix invalid files.

        Returns:
            Dictionary mapping filename -> valid status.
            True = valid, False = invalid (even after auto-fix).
        """
        results: dict[str, bool] = {}

        for filename, schema in cls.SCHEMAS.items():
            # Check both scratchpad and memory directories
            for subdir in ["scratchpad", "memory"]:
                file_path = context_dir / subdir / filename
                if not file_path.exists():
                    continue

                try:
                    if auto_fix:
                        _, was_fixed = cls.validate_file_with_auto_fix(
                            file_path, schema
                        )
                        results[str(file_path)] = True
                        if was_fixed:
                            logger.info(f"Auto-fixed {file_path}")
                    else:
                        cls.validate_file(file_path, schema)
                        results[str(file_path)] = True

                except ValidationError as e:
                    results[str(file_path)] = False
                    logger.error(f"Validation failed: {file_path}: {e}")
                except FileNotFoundError:
                    # File doesn't exist - skip
                    pass

        return results

    @classmethod
    def get_validation_errors(
        cls, file_path: Path, schema: Type[BaseModel]
    ) -> list[dict[str, str]]:
        """Get detailed validation errors for a file.

        Args:
            file_path: Path to JSON file.
            schema: Pydantic model class.

        Returns:
            List of error dictionaries with 'field' and 'message' keys.
            Empty list if file is valid.
        """
        errors: list[dict[str, str]] = []

        try:
            cls.validate_file(file_path, schema)
        except ValidationError as e:
            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])
                errors.append({"field": field, "message": err["msg"]})
        except FileNotFoundError:
            errors.append({"field": "__file__", "message": "File not found"})

        return errors


# Convenience functions for common operations


def validate_metacognition_file(
    file_path: Path, auto_fix: bool = False
) -> MetacognitiveState:
    """Validate metacognition.json file.

    Args:
        file_path: Path to metacognition.json.
        auto_fix: If True, use defaults on validation failure.

    Returns:
        Valid MetacognitiveState instance.
    """
    if auto_fix:
        state, _ = SchemaValidator.validate_file_with_auto_fix(
            file_path, MetacognitiveState
        )
        return state
    else:
        return SchemaValidator.validate_file(file_path, MetacognitiveState)


def validate_goals_file(file_path: Path, auto_fix: bool = False) -> GoalHierarchy:
    """Validate goals.json file.

    Args:
        file_path: Path to goals.json.
        auto_fix: If True, use defaults on validation failure.

    Returns:
        Valid GoalHierarchy instance.
    """
    if auto_fix:
        hierarchy, _ = SchemaValidator.validate_file_with_auto_fix(
            file_path, GoalHierarchy
        )
        return hierarchy
    else:
        return SchemaValidator.validate_file(file_path, GoalHierarchy)
