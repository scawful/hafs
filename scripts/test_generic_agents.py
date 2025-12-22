
import asyncio
from pathlib import Path
import sys
import os

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(SRC_DIR))

from agents.utility.cartographer import CartographerAgent
from agents.utility.scout import ScoutAgent

async def main():
    print(f"Repo Root: {REPO_ROOT}")
    target_dir = str(SRC_DIR / "hafs")
    
    print(f"Mapping {target_dir}")
    carto = CartographerAgent()
    res = await carto.run_task([target_dir])
    print(res["summary"])
    
    print(f"Scouting {target_dir}")
    scout = ScoutAgent()
    res2 = await scout.run_task([target_dir])
    print(res2["summary"])

if __name__ == "__main__":
    asyncio.run(main())
