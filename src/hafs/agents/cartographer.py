"""Cartographer Agent (Generic).

Maps the physical structure of the codebase.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from hafs.agents.base import BaseAgent

class CartographerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Cartographer", "Map directory structures and build dependencies.")
        self.default_roots = [str(Path.cwd())]

    async def run_task(self, roots: List[str] = None) -> Dict[str, Any]:
        """Perform a BFS mapping of the target roots."""
        target_roots = roots if roots else self.default_roots
        print(f"[{self.name}] Mapping roots: {target_roots}")
        
        map_data = {}
        summary = "## Codebase Map\n"

        for root in target_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
                
            structure = self._analyze_structure(root_path)
            
            map_data[root] = structure
            
            summary += f"\n### Root: `{root}`\n"
            summary += f"- **Subdirectories**: {len(structure['subdirs'])}\n"
            summary += "- **Key Subdirectories**:\n"
            for subdir in structure['subdirs'][:5]:
                 summary += f"  - `{subdir}`\n"

        return {"raw": map_data, "summary": summary}

    def _analyze_structure(self, root_path: Path):
        subdirs = []
        extensions = {}
        
        try:
            for item in root_path.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    subdirs.append(item.name)
                elif item.is_file():
                    ext = item.suffix
                    extensions[ext] = extensions.get(ext, 0) + 1
        except Exception as e:
            print(f"Error analyzing {root_path}: {e}")
            
        return {
            "subdirs": sorted(subdirs),
            "file_types": extensions
        }