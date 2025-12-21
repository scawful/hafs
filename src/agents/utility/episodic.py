"""Episodic Memory Agent.

Indexes chat history and agent logs into semantic memory.
"""

import json
from pathlib import Path
from typing import List, Dict
from agents.core.base import BaseAgent

class EpisodicMemoryAgent(BaseAgent):
    """The Chronicler. Remembers what happened."""

    def __init__(self):
        super().__init__("EpisodicMemoryAgent", "Summarize and index past interactions.")
        self.logs_file = self.context_root / "metrics" / "agents.jsonl"
        self.episodes_dir = self.context_root / "memory" / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self):
        # 1. Read Logs
        if not self.logs_file.exists(): return "No logs found."
        
        lines = self.logs_file.read_text().splitlines()[-500:]
        sessions = self._group_by_session(lines)
        
        new_episodes = 0
        for session_id, events in sessions.items():
            episode_file = self.episodes_dir / f"session_{session_id}.md"
            summary = await self._summarize_session(events)
            if summary:
                episode_file.write_text(summary)
                new_episodes += 1
                
        return f"Processed {len(sessions)} sessions. Created {new_episodes} summaries."

    def _group_by_session(self, lines: List[str]) -> Dict[str, List[Dict]]:
        sessions = {}
        current_session = []
        last_time = 0
        session_idx = 0
        
        for line in lines:
            try:
                evt = json.loads(line)
                # Handle timestamp parsing safely
                ts = 0
                if 'timestamp' in evt:
                    # Simple heuristic since public logs might differ
                    ts = float(evt['timestamp']) if isinstance(evt['timestamp'], (int, float)) else 0
                
                if last_time and (ts - last_time > 1800):
                    if current_session:
                        sessions[f"auto_{session_idx}"] = current_session
                        session_idx += 1
                        current_session = []
                
                current_session.append(evt)
                last_time = ts
            except: pass
            
        if current_session:
            sessions[f"auto_{session_idx}"] = current_session
            
        return sessions

    async def _summarize_session(self, events: List[Dict]) -> str:
        # Placeholder summary logic
        return f"Session Summary: {len(events)} events processed."
