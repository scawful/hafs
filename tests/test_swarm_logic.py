"""Test Public Swarm Logic."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.core.base import BaseAgent
from agents.swarm.swarm import SwarmCouncil

async def test_swarm():
    print("--- Testing Public Swarm Council ---")
    
    # Create mock agents
    mock_strat = BaseAgent("SwarmStrategist", "Plan")
    mock_rev = BaseAgent("CouncilReviewer", "Critique")
    mock_doc = BaseAgent("DeepDiveDocumenter", "Synthesize")
    mock_coll = BaseAgent("BugCollector", "Collect")
    
    # Mock their run_task methods
    async def mock_run(res):
        return res

    mock_strat.run_task = MagicMock(return_value=mock_run({"knowledge_queries": ["test"]}))
    mock_rev.run_task = MagicMock(return_value=mock_run("CONFIDENCE_SCORE: 90"))
    mock_doc.run_task = MagicMock(return_value=mock_run("# Final Report"))
    mock_coll.run_task = MagicMock(return_value=mock_run({"raw": [], "summary": "Found items"}))
    
    council = SwarmCouncil(
        {
            "strategist": mock_strat,
            "reviewer": mock_rev,
            "documenter": mock_doc,
            "primary_kb": mock_coll,
        }
    )

    print(f"Council setup with {len(council.agents_map)} agents.")
    
    # Run a session (Lite mode)
    print("Running mock session...")
    try:
        result = await council.run_session("Testing Swarm")
        print("✅ Swarm session completed.")
        report = result.get("report", "")
        assert result.get("status") == "success"
        assert "Final Report" in report
        print(f"Report Preview: {report[:100]}...")
    except Exception as e:
        print(f"❌ Swarm session failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_swarm())
