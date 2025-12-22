"""Knowledge Graph Agent.

Extracts structured relationships from unstructured markdown using LLM reasoning.
Performs entity resolution and semantic linking.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.core import BaseAgent
from core.orchestrator import ModelOrchestrator

logger = logging.getLogger(__name__)


class KnowledgeGraphAgent(BaseAgent):
    """The Architect. Builds the map of meaning."""

    def __init__(self, context_root: Optional[Path] = None):
        super().__init__("KnowledgeGraphAgent", "Extract entities and relationships from verified and discovered knowledge.")
        self.context_root = context_root or Path.home() / ".context"
        self.verified_dir = self.context_root / "knowledge" / "verified"
        self.discovered_dir = self.context_root / "knowledge" / "discovered"
        self.graph_path = self.context_root / "memory" / "knowledge_graph.json"

        # Backward compatibility / Heuristic: if global dir doesn't exist, try the legacy VerifiedContext
        if not self.verified_dir.exists() and (Path.home() / "Code" / "VerifiedContext").exists():
             self.verified_dir = Path.home() / "Code" / "VerifiedContext"

        self.orchestrator = None

    async def setup(self):
        await super().setup()
        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)

    async def build_graph(self):
        logger.info(f"[{self.name}] Building Knowledge Graph (LLM Augmented)...")

        # 1. Load existing graph if it exists
        nodes = {}
        edges = []
        if self.graph_path.exists():
            try:
                graph_data = json.loads(self.graph_path.read_text())
                nodes = graph_data.get("nodes", {})
                edges = graph_data.get("edges", [])
                logger.info(f"[{self.name}] Loaded existing graph: {len(nodes)} nodes, {len(edges)} edges.")
            except:
                pass # Start fresh if corrupt

        # 2. Backup existing graph
        if self.graph_path.exists():
            backup_path = self.graph_path.with_suffix(f".json.bak_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            shutil.copy(self.graph_path, backup_path)

        # 3. Process files
        files = []
        if self.verified_dir.exists():
            files.extend(list(self.verified_dir.glob("*.md")))
        if self.discovered_dir.exists():
            files.extend(list(self.discovered_dir.glob("*.md")))

        if not files:
            logger.info(f"[{self.name}] No markdown sources found, continuing with KB enrichment.")

        for f in files:
            content = f.read_text(errors='replace')[:12000]
            doc_node = f.name

            # Add Document Node if not present
            if doc_node not in nodes:
                nodes[doc_node] = {"type": "document", "path": str(f)}

            # --- LLM Entity Extraction & Normalization ---
            if self.orchestrator:
                extracted = await self._extract_graph_elements(f.name, content)

                # Merge Nodes
                for n in extracted.get("nodes", []):
                    nid = n.get("id")
                    if nid and nid not in nodes:
                        nodes[nid] = n

                # Merge Edges
                for e in extracted.get("edges", []):
                    if e not in edges:
                        edges.append(e)

            # Fallback / Augmentation with Regex
            self._apply_regex_heuristics(content, doc_node, nodes, edges)

        # Enrich graph with disassembly knowledge bases (if present)
        self._ingest_disassembly_kbs(nodes, edges)

        graph = {"nodes": nodes, "edges": edges}
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(json.dumps(graph, indent=2))

        logger.info(f"[{self.name}] Graph built: {len(nodes)} nodes, {len(edges)} edges.")
        return graph

    async def _extract_graph_elements(self, filename: str, text: str) -> dict[str, Any]:
        prompt = (
            f"Analyze the following documentation file: '{filename}'\n"
            "TASK: Extract a Knowledge Graph (Nodes and Edges) representing the system architecture and work streams.\n\n"
            "RULES for ENTITIES (Nodes):\n"
            "1. **Core Concepts**: High-level systems (e.g., 'Database', 'API Gateway', 'Auth Service'). Normalize names for consistency.\n"
            "2. **Artifacts**: Specific Issue IDs (#123), Code Reviews (PR 123), File Paths (src/path).\n"
            "3. **Agents/People**: Users or Agents mentioned.\n\n"
            "RULES for RELATIONSHIPS (Edges):\n"
            "1. Use strong verbs: 'blocks', 'implements', 'deprecates', 'documents', 'tests'.\n"
            "2. Link the Document itself ('{filename}') to the concepts it describes.\n\n"
            "OUTPUT FORMAT (JSON):\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": \"Canonical Name\", \"type\": \"CORE_CONCEPT|ISSUE|REVIEW|FILE_PATH\", \"meta\": {\"description\": \"...\"}}\n"
            "  ],\n"
            "  \"edges\": [\n"
            "    {\"source\": \"Node A\", \"target\": \"Node B\", \"relation\": \"blocks\"}\n"
            "  ]\n"
            "}\n"
            "Output ONLY valid JSON."
            f"\n\nTEXT CONTENT:\n{text}"
        )
        try:
            res = await self.orchestrator.generate_content(prompt, tier="fast")
            # Primitive JSON extraction from markdown block if needed
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(res)
        except:
            return {"nodes": [], "edges": []}

    def _apply_regex_heuristics(self, content: str, doc_node: str, nodes: dict, edges: list):
        # Look for Issue IDs
        issues = re.findall(r"(?i)\b(?:bug|issue)\s*#?(\d+)\b", content)
        for issue in issues:
            issue_node = f"Issue {issue}"
            if issue_node not in nodes:
                nodes[issue_node] = {"type": "issue", "id": issue}
            edges.append({"source": doc_node, "target": issue_node, "relation": "references"})

        # Look for Code Reviews
        reviews = re.findall(r"(?i)\b(?:PR|MR|review)\s*#?(\d+)\b", content)
        for review in reviews:
            review_node = f"Review {review}"
            if review_node not in nodes:
                nodes[review_node] = {"type": "review", "id": review}
            edges.append({"source": doc_node, "target": review_node, "relation": "references"})

    def _load_kb_entries(self, path: Path) -> list[dict[str, Any]]:
        """Load a list of KB entries from JSON."""
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except Exception:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [value for value in data.values() if isinstance(value, dict)]
        return []

    def _edge_key(self, edge: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("relation", "")),
        )

    def _add_edge(
        self,
        edges: list[dict[str, Any]],
        edge_set: set[tuple[str, str, str]],
        source: str,
        target: str,
        relation: str,
    ) -> bool:
        key = (source, target, relation)
        if key in edge_set:
            return False
        edges.append({"source": source, "target": target, "relation": relation})
        edge_set.add(key)
        return True

    def _ingest_disassembly_kbs(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> None:
        """Enrich graph with routine call graphs and symbol access."""
        kb_root = self.context_root / "knowledge"
        sources = [
            ("alttp", kb_root / "alttp"),
            ("oracle-of-secrets", kb_root / "oracle-of-secrets"),
        ]

        edge_set = {self._edge_key(edge) for edge in edges}
        max_routines_per_project = 300
        max_edges_per_project = 1200
        max_access_edges_per_routine = 10

        for project_name, kb_dir in sources:
            routines = self._load_kb_entries(kb_dir / "routines.json")
            if not routines:
                continue

            symbols = self._load_kb_entries(kb_dir / "symbols.json")
            symbol_by_name = {
                s.get("name"): s for s in symbols if s.get("name")
            }
            symbol_by_addr = {
                s.get("address"): s for s in symbols if s.get("address")
            }

            project_node = f"project:{project_name}"
            if project_node not in nodes:
                nodes[project_node] = {
                    "type": "project",
                    "name": project_name,
                }

            def routine_score(routine: dict[str, Any]) -> int:
                return len(routine.get("calls", [])) + len(routine.get("called_by", []))

            routines_sorted = sorted(routines, key=routine_score, reverse=True)
            selected = routines_sorted[:max_routines_per_project]

            routine_ids: dict[str, str] = {}
            for routine in selected:
                name = routine.get("name")
                if not name:
                    continue
                node_id = f"{project_name}:routine:{name}"
                routine_ids[name] = node_id
                if node_id not in nodes:
                    nodes[node_id] = {
                        "type": "routine",
                        "name": name,
                        "project": project_name,
                        "address": routine.get("address", ""),
                        "category": routine.get("category", ""),
                    }
                self._add_edge(edges, edge_set, project_node, node_id, "contains")

            edges_added = 0
            for routine in selected:
                if edges_added >= max_edges_per_project:
                    break
                name = routine.get("name")
                if not name:
                    continue
                source_id = routine_ids.get(name)
                if not source_id:
                    continue

                for target in routine.get("calls", []):
                    if edges_added >= max_edges_per_project:
                        break
                    target_id = routine_ids.get(target)
                    if target_id and self._add_edge(edges, edge_set, source_id, target_id, "calls"):
                        edges_added += 1

                access_edges = 0
                for access in routine.get("memory_access", []):
                    if edges_added >= max_edges_per_project:
                        break
                    if access_edges >= max_access_edges_per_routine:
                        break
                    sym = symbol_by_name.get(access) or symbol_by_addr.get(access)
                    if not sym:
                        continue
                    sym_name = sym.get("name")
                    if not sym_name:
                        continue
                    sym_id = f"{project_name}:symbol:{sym_name}"
                    if sym_id not in nodes:
                        nodes[sym_id] = {
                            "type": "symbol",
                            "name": sym_name,
                            "project": project_name,
                            "address": sym.get("address", ""),
                        }
                    if self._add_edge(edges, edge_set, source_id, sym_id, "accesses"):
                        edges_added += 1
                        access_edges += 1

    async def run_task(self, task: str = None):
        return await self.build_graph()


from typing import Optional
