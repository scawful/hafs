"""Knowledge Graph Agent.

Extracts structured relationships from unstructured markdown using LLM reasoning.
Performs entity resolution and semantic linking.
"""

import asyncio
import json
import re
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import shutil

from hafs.agents.base import BaseAgent
from hafs.core.orchestrator import ModelOrchestrator

class KnowledgeGraphAgent(BaseAgent):
    """The Architect. Builds the map of meaning."""

    def __init__(self, context_root: Path = None):
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
        print(f"[{self.name}] Building Knowledge Graph (LLM Augmented)...")
        
        # 1. Load existing graph if it exists
        nodes = {}
        edges = []
        if self.graph_path.exists():
            try:
                graph_data = json.loads(self.graph_path.read_text())
                nodes = graph_data.get("nodes", {})
                edges = graph_data.get("edges", [])
                print(f"[{self.name}] Loaded existing graph: {len(nodes)} nodes, {len(edges)} edges.")
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

        if not files: return
        
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

        graph = {"nodes": nodes, "edges": edges}
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(json.dumps(graph, indent=2))
        
        print(f"[{self.name}] Graph built: {len(nodes)} nodes, {len(edges)} edges.")
        return graph

    async def _extract_graph_elements(self, filename: str, text: str) -> Dict[str, Any]:
        prompt = (
            f"Analyze the following documentation file: '{filename}'\n"
            "TASK: Extract a Knowledge Graph (Nodes and Edges) representing the system architecture and work streams.\n\n"
            "RULES for ENTITIES (Nodes):\n"
            "1. **Core Concepts**: High-level systems (e.g., 'Chirp 3', 'Ganpati', 'Spanner'). Normalize names (e.g., 'Chirp3 Model' -> 'Chirp 3').\n"
            "2. **Artifacts**: Specific Bug IDs (b/123), CLs (CL 123), File Paths (//depot/path).\n"
            "3. **Agents/People**: Users or Agents mentioned (e.g. 'scawful').\n\n"
            "RULES for RELATIONSHIPS (Edges):\n"
            "1. Use strong verbs: 'blocks', 'implements', 'deprecates', 'documents', 'tests'.\n"
            "2. Link the Document itself ('{filename}') to the concepts it describes.\n\n"
            "OUTPUT FORMAT (JSON):\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": \"Canonical Name\", \"type\": \"CORE_CONCEPT|BUG|CL|FILE_PATH\", \"meta\": {\"description\": \"...\"}}\n"
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

    def _apply_regex_heuristics(self, content: str, doc_node: str, nodes: Dict, edges: List):
        # Look for Bug IDs
        bugs = re.findall(r'b/(\d+)', content)
        for bug in bugs:
            bug_node = f"Bug {bug}"
            if bug_node not in nodes:
                nodes[bug_node] = {"type": "bug", "id": bug}
            edges.append({"source": doc_node, "target": bug_node, "relation": "references"})
                
        # Look for CLs
        cls = re.findall(r'CL (\d+)', content)
        for cl in cls:
            cl_node = f"CL {cl}"
            if cl_node not in nodes:
                nodes[cl_node] = {"type": "cl", "id": cl}
            edges.append({"source": doc_node, "target": cl_node, "relation": "references"})

    async def run_task(self):
        return await self.build_graph()