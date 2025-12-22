"""Event Bus - Pub/sub system for cross-widget communication.

This module provides a centralized event bus for the TUI that enables
decoupled communication between screens, widgets, and the application.

Event Categories:
- chat: Chat messages, streaming tokens, agent responses
- agent: Agent status changes, lane events
- tool: Tool execution results, artifacts
- phase: Pipeline phase transitions
- analysis: Research analysis results (synergy, scaling, review, doc quality)
- navigation: Screen changes, modal events
- context: Context/file changes, AFS events

Usage:
    bus = EventBus()

    # Subscribe to events
    sub = bus.subscribe("chat.*", lambda e: print(e.data))

    # Publish events
    bus.publish(ChatEvent(content="Hello", agent_id="planner"))

    # Unsubscribe
    sub.unsubscribe()
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    """Event categories for routing."""
    CHAT = "chat"
    AGENT = "agent"
    TOOL = "tool"
    PHASE = "phase"
    ANALYSIS = "analysis"
    NAVIGATION = "navigation"
    CONTEXT = "context"
    METRICS = "metrics"
    SYSTEM = "system"


@dataclass
class Event:
    """Base event class for the event bus.

    All events have a category, type, timestamp, and optional data payload.
    The full event name is "{category}.{type}" for pattern matching.
    """
    category: EventCategory = EventCategory.SYSTEM
    type: str = "event"
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def name(self) -> str:
        """Full event name for pattern matching."""
        return f"{self.category.value}.{self.type}"


# Specialized Event Types (from CHAT_MODE_RENDERER_PLAN)

@dataclass
class ChatEvent(Event):
    """Chat message event."""
    content: str = ""
    role: str = "assistant"
    agent_id: str = ""
    tags: List[str] = field(default_factory=list)
    is_streaming: bool = False
    message_id: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.CHAT
        self.type = "message"
        self.data = {
            "content": self.content,
            "role": self.role,
            "agent_id": self.agent_id,
            "tags": self.tags,
            "is_streaming": self.is_streaming,
            "message_id": self.message_id,
        }


@dataclass
class StreamTokenEvent(Event):
    """Streaming token event for real-time chat display."""
    token: str = ""
    message_id: str = ""
    agent_id: str = ""
    is_final: bool = False

    def __post_init__(self):
        self.category = EventCategory.CHAT
        self.type = "stream_token"
        self.data = {
            "token": self.token,
            "message_id": self.message_id,
            "agent_id": self.agent_id,
            "is_final": self.is_final,
        }


@dataclass
class AgentStatusEvent(Event):
    """Agent status change event."""
    agent_id: str = ""
    status: Literal["thinking", "executing", "idle", "error"] = "idle"
    health: float = 1.0
    node: Optional[str] = None
    message: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.AGENT
        self.type = "status"
        self.data = {
            "agent_id": self.agent_id,
            "status": self.status,
            "health": self.health,
            "node": self.node,
            "message": self.message,
        }


@dataclass
class ToolResultEvent(Event):
    """Tool execution result event."""
    tool_name: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    success: bool = True
    artifacts: List[Path] = field(default_factory=list)
    agent_id: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.TOOL
        self.type = "result"
        self.data = {
            "tool_name": self.tool_name,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "artifacts": [str(p) for p in self.artifacts],
            "agent_id": self.agent_id,
        }


@dataclass
class PhaseEvent(Event):
    """Pipeline phase transition event."""
    phase: Literal["plan", "execute", "verify", "summarize"] = "plan"
    progress: float = 0.0
    message: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.PHASE
        self.type = "transition"
        self.data = {
            "phase": self.phase,
            "progress": self.progress,
            "message": self.message,
        }


@dataclass
class AnalysisEvent(Event):
    """Research analysis result event (from RESEARCH_ALIGNMENT_PLAN)."""
    mode: str = ""  # synergy_tom, scaling_metrics, review_quality, doc_quality
    summary: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    score: Optional[float] = None

    def __post_init__(self):
        self.category = EventCategory.ANALYSIS
        self.type = self.mode or "result"
        self.data = {
            "mode": self.mode,
            "summary": self.summary,
            "triggers": self.triggers,
            "score": self.score,
        }


@dataclass
class NavigationEvent(Event):
    """Screen navigation event."""
    action: Literal["navigate", "push", "pop", "replace"] = "navigate"
    path: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    previous_path: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.NAVIGATION
        self.type = self.action
        self.data = {
            "path": self.path,
            "params": self.params,
            "previous_path": self.previous_path,
        }


@dataclass
class ContextEvent(Event):
    """Context/file change event."""
    action: Literal["open", "save", "close", "modify", "sync"] = "open"
    path: Optional[str] = None
    project: Optional[str] = None
    changes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.category = EventCategory.CONTEXT
        self.type = self.action
        self.data = {
            "path": self.path,
            "project": self.project,
            "changes": self.changes,
        }


@dataclass
class MetricsEvent(Event):
    """Metrics update event."""
    metric_type: str = ""  # tokens, latency, cost, embedding_coverage
    value: float = 0.0
    unit: str = ""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        self.category = EventCategory.METRICS
        self.type = self.metric_type or "update"
        self.data = {
            "metric_type": self.metric_type,
            "value": self.value,
            "unit": self.unit,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }


# Subscription Management

EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], asyncio.Future]


@dataclass
class Subscription:
    """A subscription to events on the event bus.

    Subscriptions can be cancelled by calling unsubscribe().
    """
    id: str
    pattern: str
    handler: Union[EventHandler, AsyncEventHandler]
    is_async: bool = False
    priority: int = 0
    _bus: Optional["EventBus"] = field(default=None, repr=False)

    def unsubscribe(self) -> None:
        """Unsubscribe from the event bus."""
        if self._bus:
            self._bus.unsubscribe(self.id)


class EventBus:
    """Central event bus for TUI communication.

    Supports pattern-based subscriptions using glob patterns:
    - "chat.*" matches all chat events
    - "agent.status" matches only agent status events
    - "*" matches all events

    Events are dispatched in priority order (higher priority first).
    """

    def __init__(self):
        self._subscriptions: Dict[str, Subscription] = {}
        self._pattern_cache: Dict[str, Set[str]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._paused: bool = False
        self._queued_events: List[Event] = []

    def subscribe(
        self,
        pattern: str,
        handler: Union[EventHandler, AsyncEventHandler],
        priority: int = 0,
        is_async: bool = False,
    ) -> Subscription:
        """Subscribe to events matching a pattern.

        Args:
            pattern: Glob pattern to match event names (e.g., "chat.*", "agent.status")
            handler: Callback function to invoke when matching events are published
            priority: Higher priority handlers are called first (default: 0)
            is_async: If True, handler is an async function

        Returns:
            Subscription object that can be used to unsubscribe
        """
        sub_id = str(uuid.uuid4())[:8]
        subscription = Subscription(
            id=sub_id,
            pattern=pattern,
            handler=handler,
            is_async=is_async,
            priority=priority,
            _bus=self,
        )
        self._subscriptions[sub_id] = subscription
        self._pattern_cache.clear()  # Invalidate cache
        logger.debug(f"EventBus: Subscribed {sub_id} to pattern '{pattern}'")
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID.

        Args:
            subscription_id: The subscription ID to remove

        Returns:
            True if subscription was found and removed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            self._pattern_cache.clear()
            logger.debug(f"EventBus: Unsubscribed {subscription_id}")
            return True
        return False

    def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: The event to publish

        Returns:
            Number of handlers that received the event
        """
        if self._paused:
            self._queued_events.append(event)
            return 0

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Find matching subscriptions
        matching = self._get_matching_subscriptions(event.name)

        # Sort by priority (descending)
        sorted_subs = sorted(matching, key=lambda s: s.priority, reverse=True)

        # Dispatch to handlers
        count = 0
        for sub in sorted_subs:
            try:
                if sub.is_async:
                    # Schedule async handler
                    asyncio.create_task(sub.handler(event))
                else:
                    sub.handler(event)
                count += 1
            except Exception as e:
                logger.error(f"EventBus: Handler error for {sub.id}: {e}")

        logger.debug(f"EventBus: Published {event.name} to {count} handlers")
        return count

    async def publish_async(self, event: Event) -> int:
        """Publish an event and await all async handlers.

        Args:
            event: The event to publish

        Returns:
            Number of handlers that received the event
        """
        if self._paused:
            self._queued_events.append(event)
            return 0

        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        matching = self._get_matching_subscriptions(event.name)
        sorted_subs = sorted(matching, key=lambda s: s.priority, reverse=True)

        tasks = []
        sync_count = 0

        for sub in sorted_subs:
            try:
                if sub.is_async:
                    tasks.append(sub.handler(event))
                else:
                    sub.handler(event)
                    sync_count += 1
            except Exception as e:
                logger.error(f"EventBus: Handler error for {sub.id}: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return sync_count + len(tasks)

    def _get_matching_subscriptions(self, event_name: str) -> List[Subscription]:
        """Get all subscriptions matching an event name."""
        if event_name in self._pattern_cache:
            sub_ids = self._pattern_cache[event_name]
            return [self._subscriptions[sid] for sid in sub_ids if sid in self._subscriptions]

        matching_ids = set()
        for sub_id, sub in self._subscriptions.items():
            if fnmatch.fnmatch(event_name, sub.pattern):
                matching_ids.add(sub_id)

        self._pattern_cache[event_name] = matching_ids
        return [self._subscriptions[sid] for sid in matching_ids]

    def pause(self) -> None:
        """Pause event delivery. Events will be queued."""
        self._paused = True
        logger.debug("EventBus: Paused")

    def resume(self) -> int:
        """Resume event delivery and flush queued events.

        Returns:
            Number of queued events that were delivered
        """
        self._paused = False
        queued = self._queued_events
        self._queued_events = []

        for event in queued:
            self.publish(event)

        logger.debug(f"EventBus: Resumed, flushed {len(queued)} events")
        return len(queued)

    def get_history(
        self,
        pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get recent events from history.

        Args:
            pattern: Optional pattern to filter events
            limit: Maximum number of events to return

        Returns:
            List of matching events (newest first)
        """
        events = self._event_history[-limit:][::-1]

        if pattern:
            events = [e for e in events if fnmatch.fnmatch(e.name, pattern)]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_subscription_count(self, pattern: Optional[str] = None) -> int:
        """Get the number of active subscriptions.

        Args:
            pattern: Optional pattern to count specific subscriptions

        Returns:
            Number of matching subscriptions
        """
        if pattern is None:
            return len(self._subscriptions)

        return sum(
            1 for sub in self._subscriptions.values()
            if fnmatch.fnmatch(sub.pattern, pattern)
        )


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_bus
    _global_bus = None
