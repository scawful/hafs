"""Test the Trend Watcher agent."""
import asyncio
import sys
import os
from pathlib import Path

# Add source path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from agents.utility.trend_watcher import TrendWatcher
from core.config import DISCOVERED_DIR

async def main():
    print("--- Testing Trend Watcher ---")
    
    # 1. Setup Mock Data
    DISCOVERED_DIR.mkdir(parents=True, exist_ok=True)
    mock_file = DISCOVERED_DIR / "test_signal.md"
    mock_file.write_text("""
    # New Signal
    There is a growing need for a 'Fuzz Testing' agent in HAFS.
    Users are asking for automated fuzzing of their Python code.
    This seems to be a high priority request.
    """)
    print("✅ Created mock signal in Discovered knowledge.")

    # 2. Run Agent
    agent = TrendWatcher()
    await agent.setup()
    
    print("Running TrendWatcher...")
    result = await agent.run_task()
    
    print("\n--- Result ---")
    print(result)
    
    # 3. Verify
    if "Fuzz Testing" in str(result) or "fuzzing" in str(result).lower():
        print("\n✅ SUCCESS: Trend Watcher identified the new topic.")
    else:
        print("\n❌ FAILURE: Trend Watcher missed the topic.")

    # Cleanup
    mock_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())

