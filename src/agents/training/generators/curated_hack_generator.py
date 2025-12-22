"""Curated Hack Generator for allowlisted ROM hack sources.

Loads hack definitions from hafs_scawful/config/curated_hacks.toml and
generates training samples from vetted ASM files only.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class CuratedHackSourceItem(SourceItem):
    """Source item for curated hack files."""

    hack_name: str = ""
    file_path: str = ""
    authors: list[str] = field(default_factory=list)
    notes: str = ""
    code_snippet: str = ""
    org_lines: list[str] = field(default_factory=list)

    @property
    def item_id(self) -> str:
        return f"hack:{self.hack_name}:{Path(self.file_path).name}"


class CuratedHackGenerator(DataGenerator):
    """Generate training data from allowlisted hack ASM sources."""

    CONFIG_PATH = Path.home() / "Code" / "hafs_scawful" / "config" / "curated_hacks.toml"
    OVERRIDE_PATH = (
        Path.home() / "Code" / "hafs_scawful" / "config" / "curated_hacks_overrides.toml"
    )
    MAX_FILE_BYTES = 300_000
    MAX_SNIPPET_CHARS = 1200

    def __init__(self):
        super().__init__(
            name="CuratedHackGenerator",
            domain="hack_curated",
            teacher_tier="coding",
        )
        self._orchestrator = None
        self._config: dict[str, Any] = {}
        self._hack_defs: list[dict[str, Any]] = []

    @property
    def has_hacks(self) -> bool:
        return bool(self._hack_defs)

    async def setup(self):
        await super().setup()

        self._load_config()

        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    def _load_config(self) -> None:
        if not self.CONFIG_PATH.exists():
            logger.warning(f"Curated hack config not found: {self.CONFIG_PATH}")
            return

        import tomllib

        with open(self.CONFIG_PATH, "rb") as f:
            self._config = tomllib.load(f)

        if self.OVERRIDE_PATH.exists():
            try:
                with open(self.OVERRIDE_PATH, "rb") as f:
                    overrides = tomllib.load(f)
                self._apply_overrides(overrides)
                logger.info(f"Applied curated hack overrides: {self.OVERRIDE_PATH}")
            except Exception as exc:
                logger.warning(f"Failed to load curated hack overrides: {exc}")

        self._hack_defs = list(self._config.get("hack", []))

    def _apply_overrides(self, overrides: dict[str, Any]) -> None:
        if not overrides:
            return

        curated_overrides = overrides.get("curated_hacks")
        if isinstance(curated_overrides, dict):
            self._config.setdefault("curated_hacks", {}).update(
                {k: v for k, v in curated_overrides.items() if v is not None}
            )

        base_hacks = self._config.setdefault("hack", [])
        base_by_name = {
            str(h.get("name", "")).lower(): h for h in base_hacks if h.get("name")
        }

        for override in overrides.get("hack", []):
            name = str(override.get("name", "")).strip()
            if not name:
                continue

            key = name.lower()
            target = base_by_name.get(key)
            if not target:
                base_hacks.append(override)
                base_by_name[key] = override
                continue

            for field in (
                "path",
                "authors",
                "notes",
                "weight",
                "include_globs",
                "exclude_globs",
                "review_status",
            ):
                if field not in override:
                    continue
                value = override.get(field)
                if value is None:
                    continue
                if isinstance(value, list) and not value:
                    continue
                target[field] = value

    async def extract_source_items(self) -> list[CuratedHackSourceItem]:
        if not self._hack_defs:
            await self.setup()

        items: list[CuratedHackSourceItem] = []

        curated = self._config.get("curated_hacks", {})
        allowed_exts = {ext.lower() for ext in curated.get("extensions", [".asm", ".inc"])}
        exclude_globs = curated.get("exclude_globs", [])
        max_items = int(curated.get("max_items_per_hack", 250))

        for hack in self._hack_defs:
            hack_name = hack.get("name", "unknown")
            hack_path = Path(str(hack.get("path", ""))).expanduser()
            weight = float(hack.get("weight", 1.0))
            include_globs = hack.get("include_globs", []) or []
            exclude_globs_local = hack.get("exclude_globs", []) or []

            if not hack_path.exists():
                logger.warning(f"Curated hack path missing: {hack_path}")
                continue

            files = []
            for path in hack_path.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in allowed_exts:
                    continue
                if self._is_excluded(path, exclude_globs):
                    continue

                rel_path = path.relative_to(hack_path).as_posix()
                if include_globs and not self._matches_globs(rel_path, include_globs):
                    continue
                if exclude_globs_local and self._matches_globs(rel_path, exclude_globs_local):
                    continue

                files.append(path)

            if not files:
                continue

            files = sorted(files)

            # Apply global cap first
            if len(files) > max_items:
                files = files[:max_items]

            # Apply per-hack weight as a downsample ratio (deterministic)
            if weight < 1.0:
                weighted_limit = max(1, int(len(files) * weight))
                if weighted_limit < len(files):
                    rng = random.Random(hack_name)
                    files = rng.sample(files, weighted_limit)

            for path in files:
                try:
                    if path.stat().st_size > self.MAX_FILE_BYTES:
                        continue
                    content = path.read_text(errors="ignore")
                except Exception:
                    continue

                code_snippet = content[: self.MAX_SNIPPET_CHARS]
                org_lines = self._extract_org_lines(content)
                if not org_lines and not self._has_address_reference(content):
                    continue

                items.append(
                    CuratedHackSourceItem(
                        name=path.stem,
                        content=code_snippet,
                        source=hack_name,
                        hack_name=hack_name,
                        file_path=str(path),
                        authors=hack.get("authors", []),
                        notes=hack.get("notes", ""),
                        code_snippet=code_snippet,
                        org_lines=org_lines,
                    )
                )

        logger.info(f"Extracted {len(items)} curated hack files")
        return items

    def _is_excluded(self, path: Path, exclude_globs: list[str]) -> bool:
        path_str = str(path)
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in exclude_globs)

    def _matches_globs(self, rel_path: str, globs: list[str]) -> bool:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in globs)

    def _extract_org_lines(self, content: str) -> list[str]:
        lines = []
        for line in content.splitlines():
            if re.search(r"\borg\b", line, re.IGNORECASE):
                lines.append(line.strip())
            if len(lines) >= 5:
                break
        return lines

    def _has_address_reference(self, content: str) -> bool:
        pattern = re.compile(
            r"(\$[0-9A-Fa-f]{2}:[0-9A-Fa-f]{4}|\$[0-9A-Fa-f]{4,6})"
        )
        return bool(pattern.search(content))

    def get_teacher_prompt(self, item: SourceItem) -> str:
        if not isinstance(item, CuratedHackSourceItem):
            raise TypeError(f"Expected CuratedHackSourceItem, got {type(item)}")

        org_context = "\n".join(item.org_lines) if item.org_lines else "No org directives found."

        template = get_prompt(
            "agents.training.generators.curated_hack_generator.prompt",
            default=(
                "You are an expert SNES 65816 ROM hacker. Generate training data from a curated hack file.\n\n"
                "HACK: {hack_name}\n"
                "AUTHORS: {authors}\n"
                "NOTES: {notes}\n"
                "FILE: {file_path}\n"
                "ORG LINES:\n{org_lines}\n\n"
                "CODE:\n```asm\n{code}\n```\n\n"
                "Generate a JSON object with:\n"
                "1. \"instruction\": A specific question about the hack's technique or hook.\n"
                "2. \"input\": Context including ROM/WRAM addresses (use $BB:AAAA and $7E:XXXX formats).\n"
                "3. \"output\": A clear explanation of what the hack changes, hook strategy, and how to adapt it.\n\n"
                "QUALITY REQUIREMENTS:\n"
                "- Call out exact hook addresses and bank usage when present.\n"
                "- Explain vanilla behavior before the hack (if known from context).\n"
                "- Focus on teachable ROM hacking patterns, not just what the code does.\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            ),
        )

        return template.format(
            hack_name=item.hack_name,
            authors=", ".join(item.authors) if item.authors else "unknown",
            notes=item.notes or "N/A",
            file_path=item.file_path,
            org_lines=org_context,
            code=item.code_snippet,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        if not isinstance(item, CuratedHackSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content
            data = extract_json_from_response(response)
            if not data:
                return None

            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain="hack_curated",
                source=item.hack_name,
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=prompt,
                kg_entities=[item.hack_name, item.name],
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.file_path}")
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.file_path}: {e}")
            return None
