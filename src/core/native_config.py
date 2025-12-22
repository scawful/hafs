"""Native module configuration and feature detection.

Provides a centralized way to check which native features are available
and enabled. All native features are optional with Python/NumPy fallbacks.

Usage:
    from core.native_config import native_config

    # Check if a feature should use native implementation
    if native_config.use_similarity:
        # Use native SIMD cosine similarity
        pass
    else:
        # Use NumPy fallback
        pass

    # Get full status
    print(native_config.get_status())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# Try to import native module to check availability
_NATIVE_MODULE_AVAILABLE = False
_NATIVE_FEATURES: Dict[str, bool] = {}

try:
    from core._native import (
        __has_hnsw__,
        __has_quantize__,
        __has_simdjson__,
        __has_streaming__,
    )

    _NATIVE_MODULE_AVAILABLE = True
    _NATIVE_FEATURES = {
        "similarity": True,  # Always available if module loads
        "hnsw": __has_hnsw__,
        "quantize": __has_quantize__,
        "simdjson": __has_simdjson__,
        "streaming": __has_streaming__,
    }
except ImportError:
    pass


@dataclass
class NativeModuleConfig:
    """Configuration for native C++ acceleration.

    Combines compile-time availability with runtime configuration.
    """

    # Cached config (loaded lazily)
    _config: Optional[object] = field(default=None, repr=False)

    def _get_config(self):
        """Lazily load configuration."""
        if self._config is None:
            try:
                from config.loader import load_config

                self._config = load_config().native
            except Exception:
                # Use defaults if config can't be loaded
                from config.schema import NativeConfig

                self._config = NativeConfig()
        return self._config

    @property
    def module_available(self) -> bool:
        """Whether the native C++ module is built and loadable."""
        return _NATIVE_MODULE_AVAILABLE

    @property
    def enabled(self) -> bool:
        """Whether native acceleration is enabled in config."""
        return self._get_config().enabled

    @property
    def use_similarity(self) -> bool:
        """Whether to use native SIMD similarity."""
        return (
            _NATIVE_MODULE_AVAILABLE
            and self.enabled
            and self._get_config().similarity
            and _NATIVE_FEATURES.get("similarity", False)
        )

    @property
    def use_hnsw(self) -> bool:
        """Whether to use native HNSW index."""
        return (
            _NATIVE_MODULE_AVAILABLE
            and self.enabled
            and self._get_config().hnsw_index
            and _NATIVE_FEATURES.get("hnsw", False)
        )

    @property
    def use_quantization(self) -> bool:
        """Whether to use native quantization."""
        return (
            _NATIVE_MODULE_AVAILABLE
            and self.enabled
            and self._get_config().quantization
            and _NATIVE_FEATURES.get("quantize", False)
        )

    @property
    def use_simdjson(self) -> bool:
        """Whether to use SIMD-accelerated JSON parsing."""
        return (
            _NATIVE_MODULE_AVAILABLE
            and self.enabled
            and self._get_config().simdjson
            and _NATIVE_FEATURES.get("simdjson", False)
        )

    @property
    def use_streaming(self) -> bool:
        """Whether to use native streaming index."""
        return (
            _NATIVE_MODULE_AVAILABLE
            and self.enabled
            and self._get_config().streaming_index
            and _NATIVE_FEATURES.get("streaming", False)
        )

    @property
    def embedding_model(self) -> str:
        """Preferred embedding model for Ollama."""
        return self._get_config().embedding_model

    @property
    def embedding_fallback(self) -> str:
        """Fallback embedding model."""
        return self._get_config().embedding_fallback

    def get_status(self) -> Dict[str, Dict[str, bool]]:
        """Get full status of native features.

        Returns:
            Dict with 'available', 'enabled', and 'active' for each feature
        """
        cfg = self._get_config()
        features = ["similarity", "hnsw_index", "quantization", "simdjson", "streaming_index"]
        native_keys = ["similarity", "hnsw", "quantize", "simdjson", "streaming"]
        use_props = [
            self.use_similarity,
            self.use_hnsw,
            self.use_quantization,
            self.use_simdjson,
            self.use_streaming,
        ]
        cfg_vals = [
            cfg.similarity,
            cfg.hnsw_index,
            cfg.quantization,
            cfg.simdjson,
            cfg.streaming_index,
        ]

        result = {
            "module_available": _NATIVE_MODULE_AVAILABLE,
            "config_enabled": self.enabled,
            "features": {},
        }

        for feature, native_key, use_prop, cfg_val in zip(
            features, native_keys, use_props, cfg_vals
        ):
            result["features"][feature] = {
                "compiled": _NATIVE_FEATURES.get(native_key, False),
                "config_enabled": cfg_val,
                "active": use_prop,
            }

        return result

    def __repr__(self) -> str:
        if not _NATIVE_MODULE_AVAILABLE:
            return "NativeModuleConfig(module_available=False)"
        active = [k for k, v in _NATIVE_FEATURES.items() if v]
        return f"NativeModuleConfig(enabled={self.enabled}, active={active})"


# Global singleton
native_config = NativeModuleConfig()
