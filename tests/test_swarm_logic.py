"""Test Public Swarm Logic."""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.agents.base import BaseAgent
from hafs.agents.swarm import SwarmCouncil

async def test_swarm():
    print("--- Testing Public Swarm Council ---")
    
    # 1. Setup Council
    council = SwarmCouncil(scale="LOW")
    
    # Create mock agents
    mock_strat = BaseAgent("SwarmStrategist", "Plan")
    mock_rev = BaseAgent("CouncilReviewer", "Critique")
    mock_doc = BaseAgent("DeepDiveDocumenter", "Synthesize")
    mock_coll = BaseAgent("BugCollector", "Collect")
    
    # Mock their run_task methods
    async def mock_run(res):
        return res

    mock_strat.run_task = MagicMock(return_value=mock_run({"bug_query": "test"}))
    mock_rev.run_task = MagicMock(return_value=mock_run("CONFIDENCE_SCORE: 90"))
    mock_doc.run_task = MagicMock(return_value=mock_run("# Final Report"))
    mock_coll.run_task = MagicMock(return_value=mock_run({"raw": [], "summary": "Found items"}))
    
    # Inject into council
    council.strategist = mock_strat
    council.reviewer = mock_rev
    council.documenter = mock_doc
    council.agents_map = {
        "SwarmStrategist": mock_strat,
        "CouncilReviewer": mock_rev,
        "DeepDiveDocumenter": mock_doc,
        "BugCollector": mock_coll
    }
    council.agents_list = list(council.agents_map.values())

    print(f"Council setup with {len(council.agents_list)} agents.")
    
    # Run a session (Lite mode)
    print("Running mock session...")
    try:
        report = await council.run_session("Testing Swarm")
        print("✅ Swarm session completed.")
        print(f"Report Preview: {report[:100]}...")
    except Exception as e:
        print(f"❌ Swarm session failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_swarm())
