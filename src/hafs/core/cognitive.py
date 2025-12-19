"""Cognitive Layer for HAFS.

Manages the emotional state and reasoning trace of the agent system.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class CognitiveState:
    emotional_state: str = "NEUTRAL" # ANXIETY, CONFIDENCE, CURIOSITY, NEUTRAL
    confidence: float = 0.5
    last_thought: str = ""
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)

class CognitiveLayer:
    """Manages the 'internal monologue' and 'emotional state' of the swarm."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path.home() / ".context/memory/cognitive_state.json"
        self.state = self._load_state()

    def _load_state(self) -> CognitiveState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return CognitiveState(**data)
            except:
                pass
        return CognitiveState()

    def save(self):
        """Persist the cognitive state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.__dict__, f, indent=2)

    def update_state(self, emotional_state: str, confidence: float, thought: str):
        """Update the system's cognitive metrics."""
        self.state.emotional_state = emotional_state
        self.state.confidence = confidence
        self.state.last_thought = thought
        
        # Add to trace
        self.state.reasoning_trace.append({
            "timestamp": datetime.now().isoformat(),
            "emotional_state": emotional_state,
            "confidence": confidence,
            "thought": thought
        })
        
        # Limit trace size
        if len(self.state.reasoning_trace) > 100:
            self.state.reasoning_trace = self.state.reasoning_trace[-100:]
            
        self.save()

    def get_context_injection(self) -> str:
        """Returns a string describing the current cognitive state for LLM prompts."""
        return (
            f"--- COGNITIVE STATE ---\n"
            f"Emotional State: {self.state.emotional_state}\n"
            f"System Confidence: {self.state.confidence:.2f}\n"
            f"Last Thought: {self.state.last_thought}\n"
            f"-----------------------"
        )