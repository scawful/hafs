"""Code Describer Agent.

Summarizes code files, routines, and classes to generate human-readable documentation
and metadata for embedding indexing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Type

from agents.core.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class CodeUnit:
    """A discrete code symbol or routine extracted from a file."""

    name: str
    kind: str
    file_path: str = ""
    line_number: int = 0
    code: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            path = self.file_path or "unknown"
            self.id = f"{path}:{self.line_number}:{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code": self.code,
            "description": self.description,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeUnit":
        return cls(**data)


class LanguagePlugin:
    """Base class for language-specific code extraction."""

    language: str = "unknown"
    extensions: tuple[str, ...] = ()

    def extract_units(self, code: str, file_path: Optional[str] = None) -> List[CodeUnit]:
        raise NotImplementedError

    def build_prompt(self, unit: CodeUnit) -> str:
        return (
            f"Describe the {unit.kind} `{unit.name}`. "
            "Include what it does and any notable inputs/outputs."
        )


_PLUGIN_REGISTRY: Dict[str, Type[LanguagePlugin]] = {}
_EXTENSION_REGISTRY: Dict[str, str] = {}


def register_plugin(name: str, plugin: Type[LanguagePlugin]) -> None:
    """Register a language plugin by name and file extensions."""
    _PLUGIN_REGISTRY[name] = plugin
    for ext in getattr(plugin, "extensions", ()):
        normalized = ext.lower().lstrip(".")
        if normalized:
            _EXTENSION_REGISTRY[normalized] = name


def get_plugin(name: str) -> Optional[LanguagePlugin]:
    """Instantiate a registered plugin by name."""
    plugin_cls = _PLUGIN_REGISTRY.get(name)
    return plugin_cls() if plugin_cls else None


def detect_language(file_path: str | Path) -> Optional[str]:
    """Return the registered plugin name for a file extension, if any."""
    suffix = Path(file_path).suffix.lower().lstrip(".")
    if not suffix:
        return None
    return _EXTENSION_REGISTRY.get(suffix)


@dataclass
class CodeKnowledgeBase:
    """Simple JSON-backed knowledge base for extracted code units."""

    root: Path
    units: Dict[str, CodeUnit] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)

    def add_unit(self, unit: CodeUnit) -> None:
        self.units[unit.id] = unit

    def add_units(self, units: Iterable[CodeUnit]) -> None:
        for unit in units:
            self.add_unit(unit)

    def needs_update(self, file_path: str | Path) -> bool:
        current_hash = self._hash_file(file_path)
        return self.file_hashes.get(str(file_path)) != current_hash

    def mark_processed(self, file_path: str | Path) -> None:
        self.file_hashes[str(file_path)] = self._hash_file(file_path)

    def search(self, query: str, limit: int = 10) -> List[CodeUnit]:
        query_lower = query.lower()
        matches = [
            unit for unit in self.units.values()
            if query_lower in unit.name.lower() or query_lower in unit.description.lower()
        ]
        return matches[:limit]

    def save(self, filename: str = "code_kb.json") -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "units": [unit.to_dict() for unit in self.units.values()],
            "file_hashes": self.file_hashes,
        }
        target = self.root / filename
        target.write_text(json.dumps(data, indent=2))
        return target

    @classmethod
    def load(cls, root: Path, filename: str = "code_kb.json") -> "CodeKnowledgeBase":
        kb = cls(root=root)
        target = root / filename
        if not target.exists():
            return kb
        data = json.loads(target.read_text())
        for unit_data in data.get("units", []):
            unit = CodeUnit.from_dict(unit_data)
            kb.units[unit.id] = unit
        kb.file_hashes = data.get("file_hashes", {})
        return kb

    @staticmethod
    def _hash_file(file_path: str | Path) -> str:
        path = Path(file_path)
        return hashlib.md5(path.read_bytes()).hexdigest()


class CodeDescriber(BaseAgent):
    """Generates natural language descriptions of code components."""

    def __init__(self):
        super().__init__(
            "CodeDescriber",
            "Generate semantic descriptions of code components for indexing and documentation."
        )
        self.model_tier = "fast"  # Descriptions are usually simple enough for fast models

    async def describe_routine(
        self,
        name: str,
        code: str,
        context: Optional[str] = None
    ) -> str:
        """Generate a description for an assembly routine or function."""

        prompt = f"""Describe this code routine: {name}

CODE:
```
{code}
```

{f"CONTEXT: {context}" if context else ""}

Provide a concise 1-2 sentence description of what this routine does, its main inputs/outputs, and any important side effects."""

        description = await self.generate_thought(prompt)
        return description.strip()

    async def describe_symbol(
        self,
        name: str,
        context: str
    ) -> str:
        """Generate a description for a memory symbol or constant."""

        prompt = f"""Describe this codebase symbol: {name}

CONTEXT FROM CODE:
{context}

Provide a very brief (one sentence) description of what this symbol represents in the game's memory or logic."""

        description = await self.generate_thought(prompt)
        return description.strip()

    async def summarize_module(
        self,
        name: str,
        symbols: List[str],
        summary: Optional[str] = None
    ) -> str:
        """Generate a high-level summary of a code module."""

        prompt = f"""Summarize this code module: {name}

SYMBOLS/ROUTINES IN MODULE:
{", ".join(symbols[:50])}

{f"EXISTING SUMMARY: {summary}" if summary else ""}

Provide a concise paragraph summarizing the primary purpose and scope of this module within the codebase."""

        description = await self.generate_thought(prompt)
        return description.strip()

    async def run_task(self, task: Dict[str, Any]) -> str:
        """Run description task."""
        kind = task.get("kind", "routine")
        name = task.get("name", "unknown")
        content = task.get("content", "")
        context = task.get("context")

        if kind == "routine":
            return await self.describe_routine(name, content, context)
        elif kind == "symbol":
            return await self.describe_symbol(name, content)
        elif kind == "module":
            symbols = task.get("symbols", [])
            return await self.summarize_module(name, symbols, content)

        return "Unknown description task"


__all__ = [
    "CodeDescriber",
    "CodeKnowledgeBase",
    "CodeUnit",
    "LanguagePlugin",
    "register_plugin",
    "get_plugin",
    "detect_language",
]
