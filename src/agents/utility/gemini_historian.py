"""Gemini Historian Agent.

Analyzes local .gemini logs to understand project evolution and state changes.
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from agents.core.base import BaseAgent
from hafs.core.orchestrator import ModelOrchestrator

GEMINI_ROOT = Path.home() / ".gemini"
TMP_DIR = GEMINI_ROOT / "tmp"

class GeminiHistorian(BaseAgent):
    """Archaeologist for Gemini logs."""

    def __init__(self):
        super().__init__("GeminiHistorian", "Analyze conversation history to reconstruct context.")
        self.sessions = []
        self.orchestrator = ModelOrchestrator()
        self.embeddings = []
        self.metadata = []

    async def run_task(self, query: str = "recent") -> Dict[str, Any]:
        """Scan logs and answer the query."""
        print(f"[{self.name}] Scanning Gemini logs...")
        self.sessions = self._scan_sessions()
        
        if not self.sessions:
            return {"summary": "No logs found."}

        # If query is semantic, try embedding search
        if "find" in query or "search" in query or "analysis" in query:
            return await self._semantic_search(query)

        if query == "recent":
            return self._summarize_recent()
        elif "restore" in query or "state" in query:
            return self._find_state_changes(query)
        
        return {"summary": f"Found {len(self.sessions)} sessions. Use specific queries."}

    def _scan_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        if not TMP_DIR.exists(): return []

        for root, dirs, files in os.walk(TMP_DIR):
            if "chats" in dirs:
                chat_dir = Path(root) / "chats"
                for chat_file in chat_dir.glob("session-*.json"):
                    try:
                        data = json.loads(chat_file.read_text())
                        if "messages" in data:
                            sessions.append({
                                "path": str(chat_file),
                                "timestamp": data.get("startTime", ""),
                                "messages": data["messages"]
                            })
                    except: pass
        
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions

    async def _semantic_search(self, query: str) -> Dict[str, Any]:
        """Perform semantic search on user messages."""
        print(f"[{self.name}] Embedding logs for search: '{query}'...")
        
        # Build index if empty (lazy loading)
        if not self.embeddings:
            docs = []
            for s in self.sessions[:20]: # Limit to recent 20 for speed
                for m in s['messages']:
                    if m['type'] == 'user' and len(m['content']) > 10:
                        docs.append(m['content'])
                        self.metadata.append({"ts": s['timestamp'], "content": m['content']})
            
            # Batch embed? (One by one for now to be safe)
            for doc in docs:
                emb = await self.orchestrator.embed_content(doc)
                if emb: self.embeddings.append(emb)
        
        if not self.embeddings:
            return {"summary": "Failed to generate embeddings."}

        # Search
        q_emb = await self.orchestrator.embed_content(query)
        if not q_emb: return {"summary": "Failed to embed query."}
        
        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[-5:][::-1]
        
        results = []
        summary = f"## Semantic Search Results for '{query}'\n"
        for idx in top_indices:
            meta = self.metadata[idx]
            summary += f"- [{meta['ts']}] {meta['content'][:200]}...\n"
            results.append(meta)
            
        return {"summary": summary, "results": results}

    def _summarize_recent(self) -> Dict[str, Any]:
        recent = self.sessions[:5]
        summary = "## Recent Sessions\n"
        for s in recent:
            ts = s['timestamp']
            user_msgs = [m['content'] for m in s['messages'] if m['type'] == 'user']
            first_msg = user_msgs[0] if user_msgs else "No user input"
            summary += f"- **{ts}**: {first_msg[:100]}...\n"
        return {"summary": summary, "raw": recent}

    def _find_state_changes(self, query: str) -> Dict[str, Any]:
        """Finds moments where critical actions happened."""
        events = []
        keywords = ["git reset", "rm ", "restore", "port"]
        
        for sess in self.sessions:
            for msg in sess['messages']:
                content = msg.get('content', '')
                if any(k in content for k in keywords):
                    events.append({
                        "timestamp": msg.get("timestamp"),
                        "type": msg.get("type"),
                        "excerpt": content[:200]
                    })
        
        summary = f"Found {len(events)} relevant events for state analysis.\n"
        for e in events[:10]:
            summary += f"- [{e['timestamp']}] {e['type']}: {e['excerpt']}\n"
            
        return {"summary": summary, "events": events}
