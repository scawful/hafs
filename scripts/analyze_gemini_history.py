"""Gemini History Analyzer.

Parses .gemini/tmp log files to reconstruct conversation history and identify project states.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

GEMINI_ROOT = Path.home() / ".gemini"
TMP_DIR = GEMINI_ROOT / "tmp"

def scan_sessions() -> List[Dict[str, Any]]:
    """Scans all session-*.json files in .gemini/tmp."""
    sessions = []
    
    if not TMP_DIR.exists():
        return []

    # Walk through all subdirectories in tmp looking for 'chats' folder
    for root, dirs, files in os.walk(TMP_DIR):
        if "chats" in dirs:
            chat_dir = Path(root) / "chats"
            for chat_file in chat_dir.glob("session-*.json"):
                try:
                    data = json.loads(chat_file.read_text())
                    # Basic validation
                    if "messages" in data:
                        sessions.append({
                            "path": str(chat_file),
                            "timestamp": data.get("startTime", ""), # Or derive from filename
                            "message_count": len(data["messages"]),
                            "data": data
                        })
                except Exception as e:
                    print(f"Error reading {chat_file}: {e}")
                    
    # Sort by timestamp
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def analyze_state_transitions(sessions: List[Dict[str, Any]]) -> List[str]:
    """Identifies key state changes based on user prompts or tool outputs."""
    transitions = []
    
    for sess in sessions:
        msgs = sess["data"].get("messages", [])
        for i, msg in enumerate(msgs):
            content = msg.get("content", "")
            # Heuristic: Look for "restore", "reset", "port", "fix" keywords in USER messages
            if msg.get("type") == "user":
                lower_content = content.lower()
                if any(k in lower_content for k in ["restore", "reset", "port", "fix", "recover"]):
                    transitions.append(f"[{msg.get('timestamp')}] User: {content[:100]}...")
            
            # Heuristic: Look for "git reset" or "rm" in TOOL outputs
            if msg.get("type") == "model": # Gemini response
                 # Check for tool calls in the message or adjacent structure?
                 # The log format might vary. Assuming 'toolCalls' or similar.
                 pass # (Need to verify log schema for tool calls)

    return transitions

if __name__ == "__main__":
    print("--- Gemini History Analysis ---")
    all_sessions = scan_sessions()
    print(f"Found {len(all_sessions)} sessions.")
    
    transitions = analyze_state_transitions(all_sessions)
    print(f"\n--- Key Transitions ({len(transitions)}) ---")
    for t in transitions[:20]: # Show top 20
        print(t)
