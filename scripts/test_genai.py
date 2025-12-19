"""Test the Model Orchestrator and GenAI integration."""
import asyncio
import sys
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Add source path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.core.orchestrator import ModelOrchestrator

async def main():
    print("--- Testing Model Orchestrator ---")
    
    # 1. Initialize
    orch = ModelOrchestrator()
    print(f"Orchestrator Initialized. Client: {orch.client is not None}")
    
    # 2. Generate Content
    prompt = "Explain the difference between a mutex and a semaphore in one sentence."
    print(f"Prompt: {prompt}")
    
    try:
        response = await orch.generate_content(prompt, tier="fast")
        print("\n--- Response ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"❌ FAIL: {e}")

if __name__ == "__main__":
    asyncio.run(main())

