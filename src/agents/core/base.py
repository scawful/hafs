"""Base Agent Infrastructure (Public).

Provides the foundational class for all specialized agents in the HAFS swarm.
Supports dynamic context injection, cognitive protocols, and metrics.
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Public repo imports
from hafs.core.orchestrator import ModelOrchestrator

try:
    from hafs.core.cognitive import CognitiveLayer
except ImportError:
    CognitiveLayer = None

# We'll use a simple config pattern for the public repo for now
# or ideally integrate with existing hafs.config
@dataclass
class AgentMetrics:
    """Tracks agent performance and failures."""
    name: str
    failures: List[str] = field(default_factory=list)
    tool_usage: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def log_failure(self, error: str):
        self.failures.append(f"[{datetime.now()}] {error}")

    def log_tool(self, tool_name: str):
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

class BaseAgent:
    """Abstract base agent with metrics, orchestration, and context access."""

    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description
        self.metrics = AgentMetrics(name)
        
        # Shared Paths (Can be overridden by hafs.config)
        self.context_root = Path.home() / ".context"
        self.knowledge_dir = self.context_root / "knowledge"
        self.scratchpad_dir = self.context_root / "scratchpad" / "swarm"
        self.scratchpad_dir.mkdir(parents=True, exist_ok=True)
        
        # Infrastructure
        self.orchestrator = None
        self.model_tier = "fast"
        self.cognitive = CognitiveLayer() if CognitiveLayer else None
        
        # Note: Vector search and Curators would be plugins in the public repo

    async def setup(self):
        """Initialize resources."""
        api_key = os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.orchestrator = ModelOrchestrator(api_key)
        else:
            print(f"[{self.name}] Warning: No AISTUDIO_API_KEY or GEMINI_API_KEY found in environment.")

    async def generate_thought(self, prompt: str, topic: Optional[str] = None) -> str:
        """Generate a thought/response using the configured model."""
        if not self.orchestrator:
            api_key = os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self.orchestrator = ModelOrchestrator(api_key)

        # --- Dynamic Context Injection ---
        verified_context = ""
        # 1. Simple file-based context as fallback for public repo
        verified_dir = self.knowledge_dir / "verified"
        if verified_dir.exists():
            files = sorted(list(verified_dir.glob("*.md")), key=os.path.getmtime, reverse=True)
            for f in files[:3]:
                try:
                    verified_context += f"\n--- CONTEXT: {f.name} ---\n{f.read_text()[:2000]}...\n"
                except: pass

        # 2. Load Cognitive State
        cognitive_injection = ""
        emotional_directive = ""
        if self.cognitive:
            cognitive_injection = self.cognitive.get_context_injection()
            emotional_directive = self._get_emotional_directive(self.cognitive.state.emotional_state)
        
        # Compose final prompt
        base_system_context = (
            f"SYSTEM IDENTITY:\n"
            f"You are {self.name}, an autonomous agent within HAFS (Halext Agentic File System).\n"
            f"GOAL: {self.role_description}\n\n"
            "COGNITIVE PROTOCOL:\n"
            "1. CONTEXTUALIZE: Read provided context before acting.\n"
            "2. DELIBERATE: Plan your steps carefully.\n"
            "3. ACT: Execute your task precisely.\n"
        )

        system_context = (
            f"{base_system_context}\n\n"
            f"{cognitive_injection}\n"
            f"EMOTIONAL DIRECTIVE: {emotional_directive}\n\n"
            f"\nCONTEXT STACK:\n{verified_context}\n"
        )

        full_prompt = (
            f"{system_context}\n"
            f"TASK: {prompt}"
        )
        
        try:
            return await self.orchestrator.generate_content(full_prompt, tier=self.model_tier)
        except Exception as e:
            return f"Error in generate_thought: {e}"

    def _get_emotional_directive(self, state: str) -> str:
        """Returns instructions based on emotional state."""
        directives = {
            "ANXIETY": "CAUTION: Verify every assumption. Do not take destructive actions.",
            "CONFIDENCE": "MODE: High Confidence. Proceed with complex reasoning.",
            "CURIOSITY": "MODE: Exploration. Follow tangents and look for deep connections.",
            "NEUTRAL": "Standard operating procedure."
        }
        return directives.get(state, directives["NEUTRAL"])

    def save_artifact(self, title: str, content: str, kind: str = "finding"):
        """Save an artifact to the shared swarm scratchpad."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{kind}_{self.name}_{title}_{timestamp}.md"
        path = self.scratchpad_dir / filename
        path.write_text(content)
        return path

    async def run_task(self, *args, **kwargs):
        raise NotImplementedError
