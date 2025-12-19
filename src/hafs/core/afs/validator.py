"""HAFS Core Validator
Mirrors the TypeScript ContextEvaluator logic for the Python stack.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

class AFSValidator:
    def __init__(self, context_root: Path):
        self.root = context_root

    def check_integrity(self) -> Dict[str, Any]:
        """Verify the .context structure exists and is readable."""
        required_dirs = ["memory", "knowledge", "tools", "scratchpad", "history"]
        status = {"valid": True, "missing": [], "errors": []}
        
        for d in required_dirs:
            if not (self.root / d).is_dir():
                status["valid"] = False
                status["missing"].append(d)
        
        return status

    def validate_output(self, output_text: str, memory_facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Check if output_text contradicts established memory facts."""
        issues = []
        # Basic keyword-based contradiction check (placeholder for LLM-based check)
        for fact in memory_facts:
            key = fact.get("key", "").lower()
            if key in output_text.lower():
                # Example: If fact says 'No blocking calls' and output uses 'system()'
                pass
        return issues

    def get_fears(self) -> List[Dict[str, Any]]:
        """Load risks from fears.json."""
        fears_path = self.root / "memory" / "fears.json"
        if fears_path.exists():
            try:
                return json.loads(fears_path.read_text())
            except json.JSONDecodeError:
                return []
        return []
