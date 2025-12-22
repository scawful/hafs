from __future__ import annotations

import pytest

from agents.core.coordinator import AgentCoordinator
from backends.base import BackendRegistry, BaseChatBackend
from models.agent import AgentRole


class StubBackend(BaseChatBackend):
    """Backend stub that records messages and streams a canned reply."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []
        self.started = False

    @property
    def name(self) -> str:
        return "stub"

    async def start(self) -> bool:
        self.started = True
        return True

    async def stop(self) -> None:
        self.started = False

    async def send_message(self, message: str) -> None:
        self.sent_messages.append(message)

    async def stream_response(self):
        yield "stub response"

    async def inject_context(self, context: str) -> None:
        self.sent_messages.append(f"CTX: {context}")

    @property
    def is_running(self) -> bool:
        return self.started

    @property
    def is_busy(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_orchestrator_routes_and_streams() -> None:
    """AgentCoordinator should process and stream messages end-to-end."""
    BackendRegistry.clear()
    BackendRegistry.register(StubBackend)

    coordinator = AgentCoordinator(
        {"default_backend": "stub", "max_agents": 1, "enabled_backends": ["stub"]}
    )
    await coordinator.register_agent("alice", AgentRole.GENERAL)
    await coordinator.start_all_agents()

    recipient = await coordinator.route_message("hello world", sender="user")
    chunks: list[str] = []
    async for chunk in coordinator.stream_agent_response(recipient):
        chunks.append(chunk)

    from typing import cast

    lane = coordinator.get_lane(recipient)
    assert lane is not None
    # Access the private backend instance used by the lane
    backend = cast(StubBackend, lane._backend)

    assert any("hello world" in msg for msg in backend.sent_messages)
    assert "stub response" in chunks

    BackendRegistry.clear()
