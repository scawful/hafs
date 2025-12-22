"""Training data generators - generic infrastructure.

Domain-specific generators (zelda, oracle, etc.) should be in plugins like
hafs_scawful, NOT in this main repo. This module provides:

1. Generic generators that work with any domain
2. Plugin discovery system to load domain-specific generators

To add domain-specific generators, create a plugin with a generators/ directory
and a register_generators(curator) function.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.training.curator import DataCurator

logger = logging.getLogger(__name__)

# Generic generators that don't depend on specific codebases
from agents.training.generators.text_generator import TextDataGenerator, TextSourceItem
from agents.training.generators.error_generator import (
    ErrorSampleGenerator,
    ErrorSourceItem,
    MultiTeacherGenerator,
)
from agents.training.generators.history_miner import (
    HistoryMiner,
    WorkflowSourceItem,
)
from agents.training.generators.hafs_generator import (
    HafsSystemGenerator,
    HafsSourceItem,
)

# Plugin discovery paths
def _get_plugin_search_paths() -> list[Path]:
    paths = [
        Path.home() / ".config" / "hafs" / "plugins",
        Path.home() / "Code" / "hafs_scawful",
    ]

    try:
        from config.loader import load_config
        config = load_config()
        paths.extend(config.plugins.plugin_dirs)
    except Exception:
        pass

    # Preserve order, dedupe
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


PLUGIN_SEARCH_PATHS = _get_plugin_search_paths()


def discover_generator_plugins() -> list[Path]:
    """Find plugins that provide training generators.

    Returns paths to plugin directories that have a generators/ subdirectory.
    """
    plugins = []
    for search_path in PLUGIN_SEARCH_PATHS:
        if search_path.exists():
            generators_dir = search_path / "generators"
            if generators_dir.exists() and (generators_dir / "__init__.py").exists():
                plugins.append(search_path)
                logger.debug(f"Found generator plugin: {search_path}")
    return plugins


def load_plugin_generators(curator: "DataCurator") -> int:
    """Load and register generators from discovered plugins.

    Args:
        curator: The DataCurator instance to register generators with

    Returns:
        Number of generators registered
    """
    import importlib.util
    import sys

    registered = 0
    plugins = discover_generator_plugins()

    for plugin_path in plugins:
        plugin_name = plugin_path.name
        generators_init = plugin_path / "generators" / "__init__.py"

        try:
            # Add plugin to path temporarily
            if str(plugin_path.parent) not in sys.path:
                sys.path.insert(0, str(plugin_path.parent))

            # Load the generators module
            spec = importlib.util.spec_from_file_location(
                f"{plugin_name}.generators",
                generators_init
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"{plugin_name}.generators"] = module
                spec.loader.exec_module(module)

                # Call register_generators if it exists
                if hasattr(module, "register_generators"):
                    before = len(curator.list_domains())
                    module.register_generators(curator)
                    after = len(curator.list_domains())
                    registered += (after - before)
                    logger.info(f"Loaded {after - before} generators from {plugin_name}")

        except Exception as e:
            logger.warning(f"Failed to load generators from {plugin_name}: {e}")

    return registered


# Backwards compatibility - import zelda generators if they exist in plugin
# This allows existing code to keep working during the transition
def _lazy_import_zelda_generators():
    """Attempt to import zelda-specific generators from plugin for backwards compat."""
    try:
        # Try to import from hafs_scawful plugin (if available in path)
        from hafs_scawful.generators.asm_generator import AsmDataGenerator, AsmSourceItem
        from hafs_scawful.generators.cpp_generator import CppDataGenerator, CppSourceItem
        from hafs_scawful.generators.curated_hack_generator import (
            CuratedHackGenerator,
            CuratedHackSourceItem,
        )
        return {
            "AsmDataGenerator": AsmDataGenerator,
            "AsmSourceItem": AsmSourceItem,
            "CppDataGenerator": CppDataGenerator,
            "CppSourceItem": CppSourceItem,
            "CuratedHackGenerator": CuratedHackGenerator,
            "CuratedHackSourceItem": CuratedHackSourceItem,
        }
    except ImportError:
        return {}


# Lazy-loaded zelda generators for backwards compatibility
_zelda_compat = None


def __getattr__(name):
    """Lazy import zelda-specific generators from plugin."""
    global _zelda_compat
    zelda_names = {
        "AsmDataGenerator", "AsmSourceItem",
        "CppDataGenerator", "CppSourceItem",
        "CuratedHackGenerator", "CuratedHackSourceItem",
    }
    if name in zelda_names:
        if _zelda_compat is None:
            _zelda_compat = _lazy_import_zelda_generators()
        if name in _zelda_compat:
            logger.warning(
                f"Importing {name} from main repo is deprecated. "
                f"Import from hafs_scawful.generators instead."
            )
            return _zelda_compat[name]
        raise ImportError(
            f"{name} is a zelda-specific generator that should be in hafs_scawful plugin. "
            f"Install the plugin or import from hafs_scawful.generators."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Generic generators (always available)
    "TextDataGenerator",
    "TextSourceItem",
    "ErrorSampleGenerator",
    "ErrorSourceItem",
    "MultiTeacherGenerator",
    "HistoryMiner",
    "WorkflowSourceItem",
    "HafsSystemGenerator",
    "HafsSourceItem",
    # Plugin discovery
    "discover_generator_plugins",
    "load_plugin_generators",
    # Deprecated - will be removed (import from plugin instead)
    "AsmDataGenerator",
    "AsmSourceItem",
    "CppDataGenerator",
    "CppSourceItem",
    "CuratedHackGenerator",
    "CuratedHackSourceItem",
]
