"""Chat Protocol - Event-driven protocol for streaming chat messages.

This module defines the chat event protocol for real-time message streaming
in the HAFS TUI. It provides a structured way to communicate chat events
between the orchestration layer and UI components.

Message Types:
- user: Messages from the user
- assistant: AI agent responses
- system: System notifications and status updates
- tool_result: Tool execution results with metadata

Streaming Support:
- Token-by-token streaming with message_id tracking
- Final message markers for completion detection
- Support for multi-agent conversations with agent_id

Event Integration:
- ChatEvent: Complete messages
- StreamTokenEvent: Individual tokens for streaming
- ToolResultEvent: Tool execution results
- All events integrate with the EventBus for pub/sub communication

Usage:
    # Create a chat message
    msg = ChatMessage.user("Hello", agent_id="planner")

    # Stream tokens
    for token in stream:
        emit_stream_token(bus, token, message_id, agent_id)

    # Mark complete
    emit_stream_complete(bus, message_id, agent_id, full_content)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from hafs.ui.core.event_bus import (
    ChatEvent,
    EventBus,
    StreamTokenEvent,
    ToolResultEvent,
)


class MessageRole(str, Enum):
    """Message role types for chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"


class MessageStatus(str, Enum):
    """Status of a message in the chat."""
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ChatMessage:
    """A structured chat message.

    Represents a single message in the conversation with metadata
    for tracking, routing, and display.

    Attributes:
        content: The message text content
        role: Who sent the message (user, assistant, system, tool_result)
        agent_id: ID of the agent (for multi-agent chats)
        message_id: Unique identifier for this message
        timestamp: When the message was created
        status: Current status (pending, streaming, complete, error)
        tags: Optional tags for categorization/filtering
        metadata: Additional arbitrary metadata
    """
    content: str
    role: MessageRole
    agent_id: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    status: MessageStatus = MessageStatus.COMPLETE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, content: str, agent_id: str = "user", **kwargs) -> "ChatMessage":
        """Create a user message.

        Args:
            content: Message text
            agent_id: User identifier
            **kwargs: Additional message attributes

        Returns:
            ChatMessage with role=USER
        """
        return cls(
            content=content,
            role=MessageRole.USER,
            agent_id=agent_id,
            **kwargs
        )

    @classmethod
    def assistant(
        cls,
        content: str,
        agent_id: str,
        status: MessageStatus = MessageStatus.COMPLETE,
        **kwargs
    ) -> "ChatMessage":
        """Create an assistant message.

        Args:
            content: Response text
            agent_id: Agent identifier
            status: Message status (default: COMPLETE)
            **kwargs: Additional message attributes

        Returns:
            ChatMessage with role=ASSISTANT
        """
        return cls(
            content=content,
            role=MessageRole.ASSISTANT,
            agent_id=agent_id,
            status=status,
            **kwargs
        )

    @classmethod
    def system(cls, content: str, **kwargs) -> "ChatMessage":
        """Create a system message.

        Args:
            content: System message text
            **kwargs: Additional message attributes

        Returns:
            ChatMessage with role=SYSTEM
        """
        return cls(
            content=content,
            role=MessageRole.SYSTEM,
            agent_id="system",
            **kwargs
        )

    @classmethod
    def tool_result(
        cls,
        content: str,
        tool_name: str,
        agent_id: str = "",
        **kwargs
    ) -> "ChatMessage":
        """Create a tool result message.

        Args:
            content: Tool output
            tool_name: Name of the tool that was executed
            agent_id: Agent that executed the tool
            **kwargs: Additional message attributes

        Returns:
            ChatMessage with role=TOOL_RESULT
        """
        metadata = kwargs.pop("metadata", {})
        metadata["tool_name"] = tool_name

        return cls(
            content=content,
            role=MessageRole.TOOL_RESULT,
            agent_id=agent_id,
            metadata=metadata,
            **kwargs
        )

    def to_event(self) -> ChatEvent:
        """Convert to a ChatEvent for EventBus publishing.

        Returns:
            ChatEvent with this message's data
        """
        return ChatEvent(
            content=self.content,
            role=self.role.value,
            agent_id=self.agent_id,
            tags=self.tags,
            is_streaming=(self.status == MessageStatus.STREAMING),
            message_id=self.message_id,
        )

    def is_from_user(self) -> bool:
        """Check if this message is from a user."""
        return self.role == MessageRole.USER

    def is_from_agent(self) -> bool:
        """Check if this message is from an AI agent."""
        return self.role == MessageRole.ASSISTANT

    def is_system(self) -> bool:
        """Check if this is a system message."""
        return self.role == MessageRole.SYSTEM


@dataclass
class StreamingContext:
    """Context for managing an active streaming message.

    Tracks the state of a message that's being streamed token-by-token.
    Used by UI components to accumulate tokens and detect completion.

    Attributes:
        message_id: Unique identifier for the streaming message
        agent_id: Agent producing the stream
        accumulated_content: Content accumulated so far
        token_count: Number of tokens received
        started_at: When streaming began
        is_complete: Whether streaming has finished
    """
    message_id: str
    agent_id: str
    accumulated_content: str = ""
    token_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    is_complete: bool = False

    def append_token(self, token: str) -> None:
        """Append a token to the accumulated content.

        Args:
            token: Token to append
        """
        self.accumulated_content += token
        self.token_count += 1

    def mark_complete(self) -> None:
        """Mark this stream as complete."""
        self.is_complete = True

    def get_duration_ms(self) -> int:
        """Get streaming duration in milliseconds.

        Returns:
            Duration from start to now in milliseconds
        """
        delta = datetime.now() - self.started_at
        return int(delta.total_seconds() * 1000)


# Protocol helper functions for common operations

def emit_chat_message(
    bus: EventBus,
    message: ChatMessage,
) -> None:
    """Emit a complete chat message to the EventBus.

    Args:
        bus: EventBus instance
        message: ChatMessage to publish
    """
    bus.publish(message.to_event())


def emit_stream_token(
    bus: EventBus,
    token: str,
    message_id: str,
    agent_id: str,
    is_final: bool = False,
) -> None:
    """Emit a streaming token to the EventBus.

    Args:
        bus: EventBus instance
        token: Token text
        message_id: ID of the message being streamed
        agent_id: Agent producing the stream
        is_final: Whether this is the last token
    """
    bus.publish(StreamTokenEvent(
        token=token,
        message_id=message_id,
        agent_id=agent_id,
        is_final=is_final,
    ))


def emit_stream_complete(
    bus: EventBus,
    message_id: str,
    agent_id: str,
    full_content: str,
) -> None:
    """Emit a stream completion event.

    Publishes both a final token event and a complete ChatEvent.

    Args:
        bus: EventBus instance
        message_id: ID of the completed message
        agent_id: Agent that produced the message
        full_content: Complete message content
    """
    # Emit final token marker
    emit_stream_token(bus, "", message_id, agent_id, is_final=True)

    # Emit complete message
    message = ChatMessage.assistant(
        content=full_content,
        agent_id=agent_id,
        message_id=message_id,
        status=MessageStatus.COMPLETE,
    )
    emit_chat_message(bus, message)


def emit_tool_result(
    bus: EventBus,
    tool_name: str,
    stdout: str = "",
    stderr: str = "",
    duration_ms: int = 0,
    success: bool = True,
    agent_id: Optional[str] = None,
) -> None:
    """Emit a tool execution result to the EventBus.

    Args:
        bus: EventBus instance
        tool_name: Name of the executed tool
        stdout: Standard output from tool
        stderr: Standard error from tool
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
        agent_id: Agent that executed the tool
    """
    bus.publish(ToolResultEvent(
        tool_name=tool_name,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        success=success,
        agent_id=agent_id,
    ))


def create_streaming_context(agent_id: str) -> StreamingContext:
    """Create a new streaming context for an agent.

    Args:
        agent_id: Agent identifier

    Returns:
        New StreamingContext with generated message_id
    """
    return StreamingContext(
        message_id=str(uuid.uuid4())[:12],
        agent_id=agent_id,
    )
