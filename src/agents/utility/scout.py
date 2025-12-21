"""Scout Agent (Generic).

Proactively fetches context for dependencies.
"""

from pathlib import Path
from typing import List, Dict, Any
from agents.core.base import BaseAgent

class ScoutAgent(BaseAgent):
    """The Scout. Looks ahead to gather dependency context."""

    def __init__(self):
        super().__init__("Scout", "Pre-fetch context for dependencies.")

    async def run_task(self, targets: List[str] = None) -> Dict[str, Any]:
        """Fetch context for a list of file/dir targets."""
        print(f"[{self.name}] Scouting targets: {targets}")
        
        if not targets:
            return {"summary": "No targets to scout."}

        findings = {}
        for target in targets:
            path = Path(target)
            if not path.exists(): continue
            
            # 1. Search for README
            if path.is_dir():
                readme = path / "README.md"
                if readme.exists():
                    findings[target] = {"readme": str(readme)}
                    self._save_scout_report(target, "README", readme.read_text())
            
            # 2. Search for Configs (generic)
            if path.is_dir():
                configs = list(path.glob("*.toml")) + list(path.glob("*.json"))
                if configs:
                    findings.setdefault(target, {})["configs"] = [str(c) for c in configs]

        summary = f"Scouted {len(findings)} targets.\n"
        for t, data in findings.items():
            summary += f"- {t}: Found {', '.join(data.keys())}\n"
            
        return {"raw": findings, "summary": summary}

    def _save_scout_report(self, target: str, kind: str, content: str):
        """Save scouted info to discovered knowledge."""
        safe_name = target.replace("/", "_").replace("\\", "_")
        filename = f"scout_{safe_name}_{kind}.md"
        path = self.knowledge_dir / "discovered" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        header = f"# Scout Report: {target} ({kind})\n\n"
        path.write_text(header + content)
        print(f"[{self.name}] Saved {filename}")