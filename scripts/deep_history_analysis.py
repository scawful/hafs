
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from agents.utility.gemini_historian import GeminiHistorian
from core.orchestrator import ModelOrchestrator

async def run_deep_analysis():
    print("🚀 Starting Deep History Analysis...")
    h = GeminiHistorian()
    orch = ModelOrchestrator()
    
    # 1. Gather all sessions
    await h.run_task("recent") 
    sessions = h.sessions
    print(f"Found {len(sessions)} total sessions.")
    
    # 2. Search for state-changing events semantically
    queries = [
        "unifying public and internal repos",
        "deleting or resetting the codebase",
        "fixing the web dashboard syntax error",
        "porting agents to public repo",
        "restoring missing features"
    ]
    
    all_context = ""
    for q in queries:
        print(f"  - Searching: {q}")
        res = await h.run_task(q)
        all_context += f"### Query: {q}\n{res['summary']}\n\n"
        
    # 3. Synthesize with LLM
    print("🧠 Synthesizing History Report...")
    prompt = (
        "You are an AI Analyst for the HAFS project. Based on the following semantic search results from Gemini CLI logs, "
        "reconstruct the history of project states, major milestones, and critical errors (like destructive resets).\n\n"
        "GOAL: Provide a clear timeline and description of what has happened to the 'Public' and 'Internal' repos.\n\n"
        f"LOG DATA:\n{all_context}\n\n"
        "OUTPUT FORMAT: Markdown report with Timeline, Major States, and Lessons Learned."
    )
    
    report = await orch.generate_content(prompt, tier="reasoning")
    
    # 4. Save
    out_path = Path("docs/project_history_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"✅ Report saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(run_deep_analysis())
