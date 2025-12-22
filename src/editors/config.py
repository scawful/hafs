"""
hafs-lsp configuration management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import toml


class LSPConfig:
    """Load and manage LSP configuration."""

    DEFAULT_CONFIG_PATH = Path.home() / "Code/hafs/config/lsp.toml"

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"LSP config not found: {self.config_path}\n"
                f"Copy from: ~/Code/hafs/config/lsp.toml.example"
            )

        self.config = toml.load(self.config_path)

    def reload(self):
        """Reload configuration (for runtime changes)."""
        self.load()

    @property
    def enabled(self) -> bool:
        """Check if LSP is enabled."""
        return self.config.get("server", {}).get("enabled", False)

    @property
    def auto_start(self) -> bool:
        """Check if auto-start is enabled."""
        return self.config.get("server", {}).get("auto_start", False)

    @property
    def strategy(self) -> str:
        """Get model selection strategy."""
        return self.config.get("models", {}).get("strategy", "manual_trigger")

    def get_model(self, context_length: int = 0, manual: bool = False) -> str:
        """Get appropriate model based on strategy."""

        if not self.enabled:
            raise RuntimeError("hafs-lsp is disabled in config/lsp.toml")

        models = self.config.get("models", {})
        strategy = self.strategy

        # Manual trigger always uses configured model
        if manual:
            model_name = models.get("manual_trigger_model", "quality_model")
            return models.get(model_name, models.get("quality_model"))

        # Strategy-based selection
        if strategy == "fast_only":
            return models.get("fast_model", "qwen2.5-coder:1.5b")

        elif strategy == "quality_only":
            return models.get("quality_model", "qwen2.5-coder:7b-instruct-q4_K_M")

        elif strategy == "adaptive":
            threshold = self.config.get("performance", {}).get("adaptive_threshold", 200)
            if context_length > threshold:
                return models.get("quality_model")
            return models.get("fast_model")

        elif strategy == "manual_trigger":
            # Only respond to manual triggers
            if not manual:
                return None  # No auto-complete
            return models.get(models.get("manual_trigger_model", "quality_model"))

        # Default
        return models.get("fast_model", "qwen2.5-coder:1.5b")

    def get_custom_model(self, model_type: str = "fast") -> Optional[str]:
        """Get path to fine-tuned custom model."""
        models = self.config.get("models", {})
        custom_key = f"custom_{model_type}"
        return models.get(custom_key) or None

    def is_editor_enabled(self, editor: str) -> bool:
        """Check if LSP is enabled for specific editor."""
        return self.config.get("editors", {}).get(editor, False)

    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance/resource settings."""
        return self.config.get("performance", {})

    def get_context_settings(self) -> Dict[str, Any]:
        """Get ROM context settings."""
        return self.config.get("context", {})


# Singleton instance
_config: Optional[LSPConfig] = None


def get_config() -> LSPConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = LSPConfig()
    return _config


def reload_config():
    """Reload configuration from disk."""
    global _config
    if _config:
        _config.reload()
