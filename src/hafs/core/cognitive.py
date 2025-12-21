"""Cognitive Layer for HAFS.

Manages the emotional state and reasoning trace of the agent system.
Bridges legacy memory/cognitive_state.json with v0.3 scratchpad/emotions.json.
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

    def __init__(
        self,
        state_file: Optional[Path] = None,
        emotions_file: Optional[Path] = None,
    ):
        self.state_file = state_file or Path.home() / ".context/memory/cognitive_state.json"
        self.emotions_file = emotions_file or Path.home() / ".context/scratchpad/emotions.json"
        self.state = self._load_state()

    def _load_state(self) -> CognitiveState:
        state = CognitiveState()
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                state = CognitiveState(**data)
            except Exception:
                state = CognitiveState()

        if self.emotions_file.exists():
            try:
                data = json.loads(self.emotions_file.read_text())
                session = data.get("session", {}) if isinstance(data, dict) else {}
                if isinstance(session, dict):
                    confidence = session.get("confidence", state.confidence)
                    anxiety = session.get("anxiety", None)
                    mood = session.get("mood", "")
                    state.confidence = float(confidence) if confidence is not None else state.confidence
                    state.emotional_state = self._derive_emotional_state(
                        anxiety=anxiety,
                        confidence=state.confidence,
                        mood=str(mood),
                        payload=data,
                    )
            except Exception:
                pass

        return state

    @staticmethod
    def _derive_emotional_state(
        *,
        anxiety: Any,
        confidence: float,
        mood: str,
        payload: dict,
    ) -> str:
        if isinstance(anxiety, (int, float)) and anxiety >= 0.7:
            return "ANXIETY"
        if confidence >= 0.7:
            return "CONFIDENCE"
        if mood.lower() in {"curious", "curiosity"}:
            return "CURIOSITY"
        categories = payload if isinstance(payload, dict) else {}
        if "curiosity" in categories:
            return "CURIOSITY"
        return "NEUTRAL"

    def save(self):
        """Persist the cognitive state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.__dict__, f, indent=2)

        self.emotions_file.parent.mkdir(parents=True, exist_ok=True)
        emotions_payload: dict[str, Any] = {}
        if self.emotions_file.exists():
            try:
                emotions_payload = json.loads(self.emotions_file.read_text())
            except Exception:
                emotions_payload = {}

        session = emotions_payload.get("session", {}) if isinstance(emotions_payload, dict) else {}
        if not isinstance(session, dict):
            session = {}

        session.setdefault("mode", "default")
        session.setdefault("mood", self.state.emotional_state.lower())
        session["confidence"] = self.state.confidence
        session.setdefault("anxiety", max(0.0, 1.0 - self.state.confidence))

        emotions_payload.update(
            {
                "schema_version": emotions_payload.get("schema_version", "0.3"),
                "producer": emotions_payload.get("producer", {"name": "hafs", "version": "unknown"}),
                "last_updated": datetime.now().isoformat(),
                "session": session,
            }
        )

        with open(self.emotions_file, "w") as f:
            json.dump(emotions_payload, f, indent=2)

    def update(self, agent_name: str, state: dict[str, Any]) -> None:
        """Legacy update interface for core systems."""
        if not isinstance(state, dict):
            return
        emotional_state = str(state.get("emotional_state", self.state.emotional_state))
        confidence = state.get("confidence", self.state.confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = self.state.confidence
        thought = state.get("last_thought") or state.get("thought") or f"Update from {agent_name}"
        self.update_state(emotional_state, confidence, str(thought))

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
