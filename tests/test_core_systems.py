"""Test Public Core Systems."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.core.cognitive import CognitiveLayer
from hafs.core.orchestrator import ModelOrchestrator
from hafs.agents.base import BaseAgent
from hafs.agents.builder import Toolsmith

async def test_systems():
    print("--- Testing Public Core ---")
    
    # 1. Cognitive Layer
    cog = CognitiveLayer()
    cog.update("TestAgent", {"confidence": 0.9, "emotional_state": "TESTING"})
    print(f"Cognitive State: {cog.state.emotional_state}")
    
    # 2. Orchestrator
    orch = ModelOrchestrator()
    print(f"Orchestrator Client: {'Active' if orch.client else 'Inactive'}")
    
    # 3. Agents
    smith = Toolsmith()
    print(f"Agent: {smith.name}")
    
    # 4. Mock Task
    # Since we might not have API key in test env, we just check instantiation
    print("Core systems instantiated successfully.")

if __name__ == "__main__":
    asyncio.run(test_systems())
