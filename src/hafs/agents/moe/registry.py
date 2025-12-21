"""Model registry and routing table for MoE experts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".context" / "models" / "registry.toml"
DEFAULT_ROUTING_PATH = Path.home() / ".context" / "models" / "routing.toml"


def _repo_root() -> Optional[Path]:
    """Return repo root when running from source tree, if available."""
    try:
        return Path(__file__).resolve().parents[4]
    except Exception:
        return None


def _resolve_path(
    path: Optional[Path | str],
    env_var: str,
    default_path: Path,
    template_name: str,
) -> Optional[Path]:
    """Resolve a config path with env + template fallback."""
    if path:
        return Path(path).expanduser()

    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path).expanduser()

    if default_path.exists():
        return default_path

    root = _repo_root()
    if root:
        template_path = root / "docs" / "config" / template_name
        if template_path.exists():
            return template_path

    return default_path


@dataclass
class RoutingRule:
    """Single routing rule entry."""

    name: str
    experts: list[str]
    keywords: list[str]
    match: str = "any"
    priority: int = 50

    def match_score(self, text: str) -> Optional[tuple[float, list[str]]]:
        """Return match score and matched keywords, or None if no match."""
        if not self.keywords:
            return None

        normalized = text.lower()
        matches = [kw for kw in self.keywords if kw.lower() in normalized]

        if self.match == "all":
            if len(matches) != len(self.keywords):
                return None
        elif not matches:
            return None

        score = len(matches) / max(len(self.keywords), 1)
        if self.match == "all" and len(matches) == len(self.keywords):
            score = 1.0

        return score, matches


@dataclass
class RouteMatch:
    """Routing match result."""

    rule: RoutingRule
    score: float
    matched_keywords: list[str]


@dataclass
class RoutingDecision:
    """Decision returned from routing table match."""

    experts: list[str]
    confidences: list[float]
    matched_routes: list[RouteMatch]


@dataclass
class RoutingTable:
    """Routing table holding keyword-based rules."""

    routes: list[RoutingRule] = field(default_factory=list)
    default_experts: list[str] = field(default_factory=list)
    schema_version: int = 1
    updated: Optional[str] = None
    source_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "RoutingTable":
        """Load routing table from TOML."""
        resolved = _resolve_path(
            path,
            "HAFS_MODEL_ROUTING_PATH",
            DEFAULT_ROUTING_PATH,
            "routing.toml",
        )

        if not resolved or not resolved.exists():
            logger.info("Routing table not found; using defaults.")
            return cls(default_experts=[], source_path=resolved)

        try:
            with resolved.open("rb") as handle:
                data = tomllib.load(handle)
        except Exception as exc:
            logger.warning("Failed to load routing table: %s", exc)
            return cls(default_experts=[], source_path=resolved)

        metadata = data.get("metadata", {})
        default_experts = metadata.get("default_experts", [])
        root = _repo_root()
        if root:
            template_path = root / "docs" / "config" / "routing.toml"
            if template_path.exists() and resolved.resolve() == template_path.resolve():
                default_experts = []

        routes = []
        for entry in data.get("routes", []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "route")
            experts = list(entry.get("experts", []) or [])
            keywords = list(entry.get("keywords", []) or [])
            match = (entry.get("match") or "any").lower()
            priority = int(entry.get("priority", 50) or 50)
            routes.append(
                RoutingRule(
                    name=name,
                    experts=experts,
                    keywords=keywords,
                    match=match,
                    priority=priority,
                )
            )

        return cls(
            routes=routes,
            default_experts=default_experts,
            schema_version=int(metadata.get("schema_version", 1) or 1),
            updated=metadata.get("updated"),
            source_path=resolved,
        )

    def keywords_for_expert(self, expert: str) -> list[str]:
        """Aggregate keywords for a given expert from all routes."""
        keywords: list[str] = []
        for route in self.routes:
            if expert in route.experts:
                keywords.extend(route.keywords)
        return list(dict.fromkeys(keywords))

    def list_experts(self) -> list[str]:
        """List all expert IDs referenced by routes."""
        experts: list[str] = []
        for route in self.routes:
            for expert in route.experts:
                if expert not in experts:
                    experts.append(expert)
        return experts

    def match_intent(self, user_intent: str) -> Optional[RoutingDecision]:
        """Match intent against routing table rules."""
        if not self.routes:
            return None

        matches: list[RouteMatch] = []
        for rule in self.routes:
            result = rule.match_score(user_intent)
            if not result:
                continue
            score, matched = result
            matches.append(RouteMatch(rule=rule, score=score, matched_keywords=matched))

        if not matches:
            return None

        matches.sort(
            key=lambda m: (m.rule.priority, m.score, m.rule.name),
            reverse=True,
        )

        expert_scores: dict[str, float] = {}
        expert_priorities: dict[str, int] = {}

        for match in matches:
            for expert in match.rule.experts:
                expert_scores[expert] = max(expert_scores.get(expert, 0.0), match.score)
                expert_priorities[expert] = max(
                    expert_priorities.get(expert, 0),
                    match.rule.priority,
                )

        ordered = sorted(
            expert_scores.items(),
            key=lambda item: (
                expert_priorities.get(item[0], 0),
                item[1],
                item[0],
            ),
            reverse=True,
        )

        experts = [expert for expert, _ in ordered]
        confidences = [min(score, 1.0) for _, score in ordered]

        return RoutingDecision(
            experts=experts,
            confidences=confidences,
            matched_routes=matches,
        )


@dataclass
class ModelRecord:
    """Single model registry entry."""

    name: str
    display_name: str
    role: Optional[str] = None
    group: Optional[str] = None
    base: Optional[str] = None
    status: str = "planned"
    default_provider: Optional[str] = None
    context_window: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Return whether the model should be registered."""
        return self.status != "disabled"

    @property
    def inference_model(self) -> Optional[str]:
        """Return the model name to use for inference when available."""
        return self.model_name or self.base or self.name


@dataclass
class EnsembleRecord:
    """Collection of experts wired to a routing table."""

    name: str
    experts: list[str]
    routing_table: Optional[str] = None


@dataclass
class ModelRegistry:
    """Registry of planned/active expert models."""

    models: dict[str, ModelRecord] = field(default_factory=dict)
    ensembles: dict[str, EnsembleRecord] = field(default_factory=dict)
    schema_version: int = 1
    updated: Optional[str] = None
    source_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "ModelRegistry":
        """Load model registry from TOML."""
        resolved = _resolve_path(
            path,
            "HAFS_MODEL_REGISTRY_PATH",
            DEFAULT_REGISTRY_PATH,
            "model_registry.toml",
        )

        if not resolved or not resolved.exists():
            logger.info("Model registry not found; using empty registry.")
            return cls(source_path=resolved)

        try:
            with resolved.open("rb") as handle:
                data = tomllib.load(handle)
        except Exception as exc:
            logger.warning("Failed to load model registry: %s", exc)
            return cls(source_path=resolved)

        metadata = data.get("metadata", {})
        models: dict[str, ModelRecord] = {}
        ensembles: dict[str, EnsembleRecord] = {}

        for name, entry in (data.get("models", {}) or {}).items():
            if not isinstance(entry, dict):
                continue
            models[name] = ModelRecord(
                name=name,
                display_name=entry.get("display_name", name),
                role=entry.get("role"),
                group=entry.get("group"),
                base=entry.get("base"),
                status=entry.get("status", "planned"),
                default_provider=entry.get("default_provider"),
                context_window=entry.get("context_window"),
                tags=list(entry.get("tags", []) or []),
                notes=entry.get("notes"),
                model_name=entry.get("model_name") or entry.get("inference_model"),
                system_prompt=entry.get("system_prompt"),
            )

        for name, entry in (data.get("ensembles", {}) or {}).items():
            if not isinstance(entry, dict):
                continue
            ensembles[name] = EnsembleRecord(
                name=name,
                experts=list(entry.get("experts", []) or []),
                routing_table=entry.get("routing_table"),
            )

        return cls(
            models=models,
            ensembles=ensembles,
            schema_version=int(metadata.get("schema_version", 1) or 1),
            updated=metadata.get("updated"),
            source_path=resolved,
        )

    def list_models(self) -> list[str]:
        """List all model IDs."""
        return sorted(self.models.keys())

    def get(self, name: str) -> Optional[ModelRecord]:
        """Return a single model record."""
        return self.models.get(name)
