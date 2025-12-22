"""Mission Agents for autonomous research tasks.

Mission agents are goal-oriented autonomous agents that:
- Run in the background via the autonomy daemon
- Analyze codebases (alttp-gigaleak, usdasm) deeply
- Use and improve the embedding services
- Make discoveries and enhance the knowledge base
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from ulid import ULID

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from core.orchestrator_v2 import TaskTier, UnifiedOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ResearchMission:
    """Definition of a research mission for an agent."""

    mission_id: str
    objective: str  # What to discover/analyze
    kb_projects: list[str]  # Knowledge bases to query
    research_queries: list[str] = field(default_factory=list)  # Initial search queries

    # Execution settings
    depth: int = 2  # Graph traversal depth for related items
    max_items_per_query: int = 20
    interval_hours: float = 12.0  # How often to run

    # Enhancement settings
    enrichment_strategy: Literal["basic", "detailed", "cross_ref"] = "detailed"
    generate_embeddings: bool = True
    cross_reference: bool = True

    # State
    enabled: bool = True
    last_run: Optional[str] = None
    discoveries_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchMission":
        return cls(**data)


@dataclass
class ResearchDiscovery:
    """A discovery made during research."""

    discovery_id: str
    mission_id: str
    timestamp: str
    category: str  # pattern, cross_reference, anomaly, insight
    title: str
    description: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    related_symbols: list[str] = field(default_factory=list)
    confidence: float = 0.8
    actionable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MissionAgent(MemoryAwareAgent):
    """Base class for mission-oriented research agents.

    Provides:
    - Mission state management
    - Embedding search capabilities
    - Discovery logging and persistence
    - Report generation
    """

    def __init__(self, mission: ResearchMission):
        super().__init__(
            name=f"Mission_{mission.mission_id}",
            role_description=mission.objective,
        )
        self.mission = mission
        self.model_tier = "reasoning"  # Use reasoning tier for deep analysis

        # State
        self._discoveries: list[ResearchDiscovery] = []
        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._kb_cache: dict[str, Any] = {}

        # Paths
        self.missions_dir = self.context_root / "missions"
        self.missions_dir.mkdir(parents=True, exist_ok=True)
        self.mission_state_file = self.missions_dir / f"{mission.mission_id}_state.json"
        self.discoveries_file = self.missions_dir / f"{mission.mission_id}_discoveries.json"

    async def setup(self):
        await super().setup()
        self._load_state()

    def _load_state(self) -> None:
        """Load mission state from disk."""
        if self.mission_state_file.exists():
            try:
                data = json.loads(self.mission_state_file.read_text())
                self.mission.last_run = data.get("last_run")
                self.mission.discoveries_count = data.get("discoveries_count", 0)
            except Exception:
                pass

        if self.discoveries_file.exists():
            try:
                data = json.loads(self.discoveries_file.read_text())
                self._discoveries = [
                    ResearchDiscovery(**d) for d in data.get("discoveries", [])
                ]
            except Exception:
                pass

    def _save_state(self) -> None:
        """Persist mission state to disk."""
        try:
            self.mission.last_run = datetime.now().isoformat()
            self.mission_state_file.write_text(json.dumps({
                "mission_id": self.mission.mission_id,
                "last_run": self.mission.last_run,
                "discoveries_count": self.mission.discoveries_count,
                "objective": self.mission.objective,
            }, indent=2))

            self.discoveries_file.write_text(json.dumps({
                "mission_id": self.mission.mission_id,
                "discoveries": [d.to_dict() for d in self._discoveries[-100:]],  # Keep last 100
            }, indent=2))
        except Exception as e:
            logger.error(f"Failed to save mission state: {e}")

    async def _get_orchestrator(self) -> UnifiedOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = UnifiedOrchestrator()
            await self._orchestrator.initialize()
        return self._orchestrator

    async def search_embeddings(
        self,
        query: str,
        kb_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search knowledge base embeddings for related items.

        Uses semantic search with embeddings if available, falls back to
        keyword matching otherwise.
        """
        kb_dir = self.context_root / "knowledge" / kb_name
        embeddings_dir = kb_dir / "embeddings"
        index_file = kb_dir / "embedding_index.json"

        if not embeddings_dir.exists():
            logger.warning(f"No embeddings directory for {kb_name}")
            return []

        # Try semantic search first
        results = []
        try:
            orchestrator = await self._get_orchestrator()
            query_embedding = await orchestrator.embed(query)

            if query_embedding and index_file.exists():
                index = json.loads(index_file.read_text())

                for item_id, emb_file in list(index.items())[:500]:  # Limit scan
                    emb_path = embeddings_dir / emb_file
                    if not emb_path.exists():
                        continue

                    try:
                        emb_data = json.loads(emb_path.read_text())
                        embedding = emb_data.get("embedding", [])
                        if not embedding:
                            continue

                        score = self._cosine_similarity(query_embedding, embedding)
                        if score > 0.4:  # Lowered threshold
                            results.append({
                                "id": item_id,
                                "score": score,
                                "text": emb_data.get("text", emb_data.get("text_preview", "")),
                                "file": emb_file,
                            })
                    except Exception:
                        continue

        except Exception as e:
            logger.warning(f"Semantic search failed, using keyword fallback: {e}")

        # Fallback to keyword matching if semantic search found nothing
        if not results:
            logger.info(f"Using keyword search for query: {query}")
            query_terms = query.lower().split()

            for emb_file in list(embeddings_dir.glob("*.json"))[:500]:
                try:
                    emb_data = json.loads(emb_file.read_text())
                    item_id = emb_data.get("id", "")
                    text = emb_data.get("text", emb_data.get("text_preview", "")).lower()

                    # Score based on keyword matches
                    matches = sum(1 for term in query_terms if term in text or term in item_id.lower())
                    if matches > 0:
                        score = matches / len(query_terms)
                        results.append({
                            "id": item_id,
                            "score": score,
                            "text": emb_data.get("text", emb_data.get("text_preview", "")),
                            "file": emb_file.name,
                        })
                except Exception:
                    continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def load_knowledge_base(self, kb_name: str) -> dict[str, Any]:
        """Load a knowledge base's symbols and routines."""
        if kb_name in self._kb_cache:
            return self._kb_cache[kb_name]

        kb_dir = self.context_root / "knowledge" / kb_name
        symbols_file = kb_dir / "symbols.json"
        routines_file = kb_dir / "routines.json"

        kb_data = {"symbols": {}, "routines": {}}

        if symbols_file.exists():
            try:
                symbols = json.loads(symbols_file.read_text())
                kb_data["symbols"] = {s["name"]: s for s in symbols} if isinstance(symbols, list) else symbols
            except Exception:
                pass

        if routines_file.exists():
            try:
                routines = json.loads(routines_file.read_text())
                kb_data["routines"] = {r["name"]: r for r in routines} if isinstance(routines, list) else routines
            except Exception:
                pass

        self._kb_cache[kb_name] = kb_data
        return kb_data

    def add_discovery(
        self,
        category: str,
        title: str,
        description: str,
        evidence: Optional[list[dict]] = None,
        related_symbols: Optional[list[str]] = None,
        confidence: float = 0.8,
    ) -> ResearchDiscovery:
        """Record a new discovery."""
        discovery = ResearchDiscovery(
            discovery_id=str(ULID()),
            mission_id=self.mission.mission_id,
            timestamp=datetime.now().isoformat(),
            category=category,
            title=title,
            description=description,
            evidence=evidence or [],
            related_symbols=related_symbols or [],
            confidence=confidence,
        )

        self._discoveries.append(discovery)
        self.mission.discoveries_count += 1
        return discovery

    async def enhance_embeddings(
        self,
        items: list[dict[str, Any]],
        kb_name: str,
    ) -> dict[str, int]:
        """Generate or update embeddings for items."""
        kb_dir = self.context_root / "knowledge" / kb_name
        embeddings_dir = kb_dir / "embeddings"
        index_file = kb_dir / "embedding_index.json"

        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index
        existing_index = {}
        if index_file.exists():
            try:
                existing_index = json.loads(index_file.read_text())
            except Exception:
                pass

        orchestrator = await self._get_orchestrator()
        created = 0
        skipped = 0

        for item in items:
            item_id = item.get("id", item.get("name", ""))
            text = item.get("text", item.get("description", ""))

            if not item_id or not text:
                skipped += 1
                continue

            # Check if already exists
            if item_id in existing_index:
                skipped += 1
                continue

            try:
                embedding = await orchestrator.embed(text)
                if embedding:
                    # Save embedding file
                    file_hash = hashlib.md5(item_id.encode()).hexdigest()[:12]
                    emb_file = f"{file_hash}.json"
                    emb_path = embeddings_dir / emb_file

                    emb_path.write_text(json.dumps({
                        "id": item_id,
                        "text": text,
                        "embedding": embedding,
                    }))

                    existing_index[item_id] = emb_file
                    created += 1
            except Exception as e:
                logger.warning(f"Failed to embed {item_id}: {e}")
                skipped += 1

        # Save updated index
        if created > 0:
            index_file.write_text(json.dumps(existing_index, indent=2))

        return {"created": created, "skipped": skipped}

    async def run_task(self) -> LoopReport:
        """Execute the research mission. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run_task()")


class ALTTPResearchAgent(MissionAgent):
    """Research agent specialized for ALTTP codebase analysis.

    Capabilities:
    - Deep analysis of 65816 ASM patterns
    - Symbol and routine relationship discovery
    - Memory access pattern analysis
    - Cross-referencing with gigaleak/Oracle
    """

    async def run_task(self) -> LoopReport:
        """Execute ALTTP research mission."""
        findings: list[str] = []
        metrics = {
            "queries_run": 0,
            "items_found": 0,
            "discoveries": 0,
            "embeddings_created": 0,
        }

        # Phase 1: Semantic search across queries
        all_results = []
        for query in self.mission.research_queries:
            for kb_name in self.mission.kb_projects:
                results = await self.search_embeddings(query, kb_name, limit=self.mission.max_items_per_query)
                all_results.extend([(kb_name, r) for r in results])
                metrics["queries_run"] += 1

        metrics["items_found"] = len(all_results)

        if not all_results:
            findings.append("No matching items found for research queries.")
        else:
            # Phase 2: Analyze patterns in results
            patterns = await self._analyze_patterns(all_results)
            findings.extend(patterns)

            # Phase 3: Cross-reference if enabled
            if self.mission.cross_reference and len(self.mission.kb_projects) > 1:
                cross_refs = await self._find_cross_references(all_results)
                if cross_refs:
                    findings.append(f"\n### Cross-References Found: {len(cross_refs)}")
                    for ref in cross_refs[:10]:
                        findings.append(f"- {ref}")

            # Phase 4: Generate enhanced embeddings for discoveries
            if self.mission.generate_embeddings and self._discoveries:
                items_to_embed = [
                    {"id": d.discovery_id, "text": f"{d.title}: {d.description}"}
                    for d in self._discoveries[-20:]
                ]
                stats = await self.enhance_embeddings(items_to_embed, self.mission.kb_projects[0])
                metrics["embeddings_created"] = stats.get("created", 0)

        metrics["discoveries"] = len(self._discoveries)

        # Remember findings
        await self.remember(
            content="\n".join(findings)[:500],
            memory_type="discovery",
            context={
                "mission_id": self.mission.mission_id,
                "metrics": metrics,
            },
            importance=0.8,
        )

        # Save state
        self._save_state()

        # Build report
        body = f"""## Mission: {self.mission.objective}

### Research Summary
- Queries executed: {metrics['queries_run']}
- Items analyzed: {metrics['items_found']}
- New discoveries: {metrics['discoveries']}
- Embeddings created: {metrics['embeddings_created']}

### Findings
{"".join(findings) if findings else "No significant findings in this run."}

### Recent Discoveries
"""
        for d in self._discoveries[-5:]:
            body += f"\n**{d.title}** ({d.category}, confidence: {d.confidence:.0%})\n"
            body += f"{d.description[:200]}...\n"

        return LoopReport(
            title=f"ALTTP Research: {self.mission.mission_id}",
            body=body,
            tags=["mission", "alttp", self.mission.mission_id],
            metrics=metrics,
        )

    async def _analyze_patterns(self, results: list[tuple[str, dict]]) -> list[str]:
        """Analyze patterns in search results using LLM."""
        findings = []

        if len(results) < 3:
            return findings

        # Group by score ranges (adjusted for typical embedding similarity)
        high_confidence = [r for _, r in results if r["score"] > 0.6]
        medium_confidence = [r for _, r in results if 0.5 < r["score"] <= 0.6]

        if high_confidence:
            findings.append(f"\n### High Confidence Matches ({len(high_confidence)})")

            # Use LLM to analyze patterns
            sample_texts = [r["text"][:200] for r in high_confidence[:10]]
            prompt = f"""Analyze these ALTTP assembly code items and identify:
1. Common patterns or themes
2. Potential relationships between items
3. Interesting technical insights

Items:
{json.dumps(sample_texts, indent=2)}

Provide 2-3 key observations in bullet points.
"""
            try:
                analysis = await self.generate_thought(prompt)
                findings.append(analysis)

                # Record as discovery if interesting analysis was generated
                if len(analysis) > 100:  # Meaningful analysis generated
                    self.add_discovery(
                        category="pattern",
                        title=f"Analysis: {self.mission.research_queries[0] if self.mission.research_queries else 'research'}",
                        description=analysis[:800],
                        related_symbols=[r.get("id", "") for r in high_confidence[:5]],
                        confidence=0.75,
                    )
            except Exception as e:
                findings.append(f"Analysis error: {e}")

        if medium_confidence:
            findings.append(f"\n### Medium Confidence Matches ({len(medium_confidence)})")
            for r in medium_confidence[:5]:
                findings.append(f"- {r['id']}: {r['text'][:100]}...")

        return findings

    async def _find_cross_references(self, results: list[tuple[str, dict]]) -> list[str]:
        """Find cross-references between knowledge bases."""
        cross_refs = []

        # Group results by KB
        by_kb: dict[str, list[dict]] = {}
        for kb_name, r in results:
            by_kb.setdefault(kb_name, []).append(r)

        if len(by_kb) < 2:
            return cross_refs

        # Compare items across KBs
        kb_names = list(by_kb.keys())
        for i, kb1 in enumerate(kb_names):
            for kb2 in kb_names[i + 1:]:
                for item1 in by_kb[kb1][:20]:
                    for item2 in by_kb[kb2][:20]:
                        # Check for name similarity
                        id1 = item1.get("id", "").lower()
                        id2 = item2.get("id", "").lower()

                        if id1 and id2:
                            # Exact match
                            if id1 == id2:
                                cross_refs.append(f"{kb1}/{id1} == {kb2}/{id2}")
                                self.add_discovery(
                                    category="cross_reference",
                                    title=f"Cross-reference: {id1}",
                                    description=f"Found matching symbol in {kb1} and {kb2}",
                                    related_symbols=[id1, id2],
                                    confidence=0.95,
                                )
                            # Partial match
                            elif id1 in id2 or id2 in id1:
                                cross_refs.append(f"{kb1}/{id1} ~ {kb2}/{id2}")

        return cross_refs


class GigaleakAnalysisAgent(MissionAgent):
    """Research agent for analyzing ALTTP gigaleak data.

    Focuses on:
    - Comparing gigaleak to vanilla ALTTP
    - Finding development artifacts
    - Discovering unused or debug code
    """

    async def run_task(self) -> LoopReport:
        """Execute gigaleak analysis mission."""
        findings = []
        metrics = {
            "items_analyzed": 0,
            "differences_found": 0,
            "discoveries": 0,
        }

        # Load knowledge bases
        await self.load_knowledge_base("alttp")

        # Research queries for gigaleak analysis
        analysis_queries = self.mission.research_queries or [
            "debug routine",
            "unused function",
            "development artifact",
            "prototype code",
            "disabled feature",
        ]

        all_results = []
        for query in analysis_queries:
            for kb_name in self.mission.kb_projects:
                results = await self.search_embeddings(query, kb_name, limit=15)
                all_results.extend(results)
                metrics["items_analyzed"] += len(results)

        if all_results:
            # Analyze for development artifacts
            findings.append("### Potential Development Artifacts")

            for result in all_results[:20]:
                text = result.get("text", "").lower()
                item_id = result.get("id", "")

                # Look for debug/unused patterns
                if any(kw in text for kw in ["debug", "unused", "disabled", "test", "prototype"]):
                    findings.append(f"- **{item_id}**: {result['text'][:150]}...")
                    metrics["differences_found"] += 1

                    self.add_discovery(
                        category="artifact",
                        title=f"Development artifact: {item_id}",
                        description=result["text"][:500],
                        confidence=result["score"],
                    )

        metrics["discoveries"] = len(self._discoveries)

        await self.remember(
            content=f"Gigaleak analysis: {metrics['differences_found']} artifacts found",
            memory_type="discovery",
            context=metrics,
            importance=0.8,
        )

        self._save_state()

        body = f"""## Gigaleak Analysis Report

### Summary
- Items analyzed: {metrics['items_analyzed']}
- Development artifacts found: {metrics['differences_found']}
- Total discoveries: {metrics['discoveries']}

### Findings
{chr(10).join(findings) if findings else "No significant artifacts found in this run."}

### Discovery Categories
"""
        categories = {}
        for d in self._discoveries:
            categories[d.category] = categories.get(d.category, 0) + 1
        for cat, count in categories.items():
            body += f"- {cat}: {count}\n"

        return LoopReport(
            title="Gigaleak Analysis",
            body=body,
            tags=["mission", "gigaleak", "analysis"],
            metrics=metrics,
        )


# Pre-defined missions
DEFAULT_MISSIONS = [
    ResearchMission(
        mission_id="alttp_sprite_patterns",
        objective="Analyze sprite behavior and rendering patterns in ALTTP",
        kb_projects=["alttp"],
        research_queries=[
            "sprite movement",
            "sprite animation",
            "sprite collision",
            "sprite draw",
            "OAM buffer",
        ],
        interval_hours=12.0,
        enrichment_strategy="detailed",
    ),
    ResearchMission(
        mission_id="alttp_memory_mapping",
        objective="Map WRAM usage patterns and memory organization",
        kb_projects=["alttp"],
        research_queries=[
            "WRAM allocation",
            "memory region",
            "state machine",
            "game state",
            "save data",
        ],
        interval_hours=24.0,
        enrichment_strategy="detailed",
    ),
    ResearchMission(
        mission_id="alttp_cross_reference",
        objective="Find relationships between ALTTP and Oracle of Secrets",
        kb_projects=["alttp", "oracle"],
        research_queries=[
            "dungeon routine",
            "overworld handler",
            "item logic",
            "player movement",
        ],
        interval_hours=24.0,
        cross_reference=True,
    ),
]


def get_mission_agent(mission: ResearchMission) -> MissionAgent:
    """Factory to create appropriate agent for a mission."""
    if "gigaleak" in mission.mission_id.lower():
        return GigaleakAnalysisAgent(mission)
    elif "alttp" in mission.kb_projects or "alttp" in mission.mission_id.lower():
        return ALTTPResearchAgent(mission)
    else:
        return ALTTPResearchAgent(mission)  # Default
