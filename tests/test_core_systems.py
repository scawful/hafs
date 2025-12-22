"""Test Public Core Systems."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.cognitive import CognitiveLayer
from core.orchestrator import ModelOrchestrator
from agents.core.base import BaseAgent
from agents.builder import Toolsmith

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
