"""Test how SwarmCouncil handles failing agents."""
import asyncio
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.agents.base import BaseAgent
from hafs.agents.swarm import SwarmCouncil

class FailingAgent(BaseAgent):
    def __init__(self):
        super().__init__("FailingAgent", "I always fail.")

    async def setup(self):
        await super().setup()

    async def run_task(self, *args, **kwargs):
        raise ValueError("Intentional Failure")

async def main():
    print("--- Testing Swarm Failure Handling ---")
    
    failing_agent = FailingAgent()
    council = SwarmCouncil({"FailingAgent": failing_agent})
    await council.setup()

    # We need to manually invoke the parallel runner or a session that uses it
    # Since run_session is complex, let's call _run_parallel_tasks directly
    
    tasks = {"test_fail": failing_agent.run_task()}
    
    print("Running task...")
    results = await council._run_parallel_tasks(tasks)
    
    print("\nResults:")
    print(results)
    
    if "test_fail" in results and "summary" in results["test_fail"]:
        summary = results["test_fail"]["summary"]
        if "Intentional Failure" in summary:
             print("✅ SUCCESS: Failure was caught and reported.")
        else:
             print(f"❌ FAILURE: Unexpected summary: {summary}")
    else:
        print("❌ FAILURE: Result structure is wrong.")

if __name__ == "__main__":
    asyncio.run(main())
