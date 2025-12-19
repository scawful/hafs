"""Halext-org backend API client.

Provides client for communicating with the halext-org FastAPI backend,
including authentication, task/event management, AI gateway, context sync,
and WebSocket conversations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TokenPair:
    """JWT token pair for authentication."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None


@dataclass
class Task:
    """Task model from halext-org."""
    id: int
    title: str
    description: Optional[str] = None
    status: str = "pending"
    priority: int = 2
    due_date: Optional[datetime] = None
    labels: list[str] = field(default_factory=list)


@dataclass
class Event:
    """Event model from halext-org."""
    id: int
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    description: Optional[str] = None


@dataclass
class HivemindEntry:
    """Hivemind knowledge entry."""
    id: int
    key: str
    value: str
    category: str
    status: str = "golden"
    confidence: float = 1.0


@dataclass
class CognitiveState:
    """Agent cognitive state."""
    conversation_id: int
    agent_id: int
    spin_count: int = 0
    spin_pattern: Optional[str] = None
    cognitive_load: float = 0.0
    strategy: str = "explore"
    uncertainties: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


class HalextOrgClient:
    """Client for halext-org backend API.

    Provides methods for:
    - Authentication (login, token refresh)
    - Task and event management
    - AI gateway (chat, embeddings)
    - Context and hivemind sync
    - WebSocket conversations
    - Cognitive state management

    Example:
        client = HalextOrgClient("https://halext.org/api")
        await client.login("username", "password")

        # Get tasks
        tasks = await client.get_tasks(status="pending")

        # Use AI gateway
        response = await client.ai_chat([
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        base_url: str = "https://halext.org/api",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the client.

        Args:
            base_url: API base URL.
            username: Optional username for auto-login.
            password: Optional password for auto-login.
        """
        self.base_url = base_url.rstrip("/")
        self._username = username or os.environ.get("HALEXT_USERNAME")
        self._password = password or os.environ.get("HALEXT_PASSWORD")
        self._tokens: Optional[TokenPair] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        """Get headers with authorization if logged in."""
        headers = {"Content-Type": "application/json"}
        if self._tokens:
            headers["Authorization"] = f"Bearer {self._tokens.access_token}"
        return headers

    async def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> TokenPair:
        """Login to get access tokens.

        Args:
            username: Username (or use instance default).
            password: Password (or use instance default).

        Returns:
            Token pair with access and refresh tokens.
        """
        await self._ensure_session()

        user = username or self._username
        pwd = password or self._password

        if not user or not pwd:
            raise ValueError("Username and password required")

        data = {"username": user, "password": pwd}

        async with self._session.post(
            f"{self.base_url}/token/login",
            data=data,
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Login failed: {error}")

            result = await resp.json()
            self._tokens = TokenPair(
                access_token=result["access_token"],
                refresh_token=result["refresh_token"],
                token_type=result.get("token_type", "bearer"),
            )
            logger.info(f"Logged in as {user}")
            return self._tokens

    async def refresh_token(self) -> str:
        """Refresh the access token.

        Returns:
            New access token.
        """
        if not self._tokens:
            raise RuntimeError("Not logged in")

        await self._ensure_session()

        async with self._session.post(
            f"{self.base_url}/token/refresh",
            headers={"Authorization": f"Bearer {self._tokens.refresh_token}"},
        ) as resp:
            if resp.status != 200:
                # Token expired, need full login
                return await self.login()

            result = await resp.json()
            self._tokens.access_token = result["access_token"]
            return self._tokens.access_token

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make an authenticated request.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            **kwargs: Additional request kwargs.

        Returns:
            JSON response.
        """
        await self._ensure_session()

        if "headers" not in kwargs:
            kwargs["headers"] = self._headers()

        url = f"{self.base_url}{endpoint}"

        async with self._session.request(method, url, **kwargs) as resp:
            if resp.status == 401:
                # Try refreshing token
                await self.refresh_token()
                kwargs["headers"] = self._headers()
                async with self._session.request(method, url, **kwargs) as retry:
                    retry.raise_for_status()
                    return await retry.json()

            resp.raise_for_status()
            return await resp.json()

    # ========== Tasks API ==========

    async def get_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        limit: int = 50,
    ) -> list[Task]:
        """Get tasks.

        Args:
            status: Filter by status.
            priority: Filter by priority.
            limit: Max tasks to return.

        Returns:
            List of tasks.
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority

        result = await self._request("GET", "/tasks/", params=params)

        return [
            Task(
                id=t["id"],
                title=t["title"],
                description=t.get("description"),
                status=t.get("status", "pending"),
                priority=t.get("priority", 2),
                labels=t.get("labels", []),
            )
            for t in result
        ]

    async def create_task(
        self,
        title: str,
        description: Optional[str] = None,
        priority: int = 2,
        labels: Optional[list[str]] = None,
    ) -> Task:
        """Create a new task.

        Args:
            title: Task title.
            description: Optional description.
            priority: Priority (1-5).
            labels: Optional labels.

        Returns:
            Created task.
        """
        data = {
            "title": title,
            "description": description,
            "priority": priority,
            "labels": labels or [],
        }

        result = await self._request("POST", "/tasks/", json=data)

        return Task(
            id=result["id"],
            title=result["title"],
            description=result.get("description"),
            priority=result.get("priority", 2),
        )

    # ========== Events API ==========

    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[Event]:
        """Get events in date range.

        Args:
            start_date: Range start.
            end_date: Range end.

        Returns:
            List of events.
        """
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = await self._request("GET", "/events/", params=params)

        return [
            Event(
                id=e["id"],
                title=e["title"],
                start_time=datetime.fromisoformat(e["start_time"]),
                end_time=datetime.fromisoformat(e["end_time"]) if e.get("end_time") else None,
                location=e.get("location"),
                description=e.get("description"),
            )
            for e in result
        ]

    # ========== AI Gateway ==========

    async def ai_chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """Send chat to AI gateway.

        Args:
            messages: Chat messages.
            model: Optional model override.

        Returns:
            AI response text.
        """
        data = {"messages": messages}
        if model:
            data["model"] = model

        result = await self._request("POST", "/ai/chat", json=data)
        return result.get("content", "")

    async def ai_chat_stream(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from AI gateway.

        Args:
            messages: Chat messages.
            model: Optional model override.

        Yields:
            Response chunks.
        """
        await self._ensure_session()

        data = {"messages": messages}
        if model:
            data["model"] = model

        async with self._session.post(
            f"{self.base_url}/ai/chat/stream",
            json=data,
            headers=self._headers(),
        ) as resp:
            async for line in resp.content:
                if line:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: "):
                        yield text[6:]

    async def ai_embed(self, text: str) -> list[float]:
        """Generate embeddings.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        result = await self._request(
            "POST",
            "/ai/embeddings",
            json={"text": text},
        )
        return result.get("embedding", [])

    async def ai_models(self) -> list[dict[str, Any]]:
        """Get available AI models.

        Returns:
            List of model info.
        """
        return await self._request("GET", "/ai/models")

    # ========== Context & Hivemind ==========

    async def sync_context(self, context: dict[str, Any]) -> None:
        """Sync context to halext-org.

        Args:
            context: Context data to sync.
        """
        await self._request("POST", "/context/sync", json=context)

    async def get_hivemind_golden(self) -> list[HivemindEntry]:
        """Get golden hivemind entries.

        Returns:
            List of golden knowledge entries.
        """
        result = await self._request("GET", "/context/hivemind/golden")

        return [
            HivemindEntry(
                id=e["id"],
                key=e["key"],
                value=e["value"],
                category=e.get("category", "general"),
                status=e.get("status", "golden"),
                confidence=e.get("confidence", 1.0),
            )
            for e in result
        ]

    async def get_facts(self) -> dict[str, Any]:
        """Get user facts.

        Returns:
            Facts dictionary.
        """
        return await self._request("GET", "/context/facts")

    # ========== Conversations (WebSocket) ==========

    async def connect_conversation(self, conversation_id: int) -> None:
        """Connect to a conversation via WebSocket.

        Args:
            conversation_id: Conversation to connect to.
        """
        await self._ensure_session()

        if not self._tokens:
            raise RuntimeError("Must be logged in to connect to conversation")

        ws_url = self.base_url.replace("http", "ws")
        self._ws = await self._session.ws_connect(
            f"{ws_url}/ws/{conversation_id}",
            headers={"Authorization": f"Bearer {self._tokens.access_token}"},
        )
        logger.info(f"Connected to conversation {conversation_id}")

    async def send_ws_message(self, content: str) -> None:
        """Send message via WebSocket.

        Args:
            content: Message content.
        """
        if not self._ws:
            raise RuntimeError("Not connected to conversation")

        await self._ws.send_json({
            "type": "message",
            "content": content,
        })

    async def receive_ws_message(self) -> Optional[dict[str, Any]]:
        """Receive message from WebSocket.

        Returns:
            Message data or None if closed.
        """
        if not self._ws:
            return None

        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            return json.loads(msg.data)
        elif msg.type == aiohttp.WSMsgType.CLOSED:
            self._ws = None
            return None

        return None

    async def disconnect_conversation(self) -> None:
        """Disconnect from conversation."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ========== Cognitive State ==========

    async def get_cognitive_state(
        self,
        conversation_id: int,
        agent_id: int,
    ) -> CognitiveState:
        """Get agent cognitive state.

        Args:
            conversation_id: Conversation ID.
            agent_id: Agent ID.

        Returns:
            Cognitive state.
        """
        result = await self._request(
            "GET",
            f"/conversations/{conversation_id}/agents/{agent_id}/cognitive",
        )

        return CognitiveState(
            conversation_id=conversation_id,
            agent_id=agent_id,
            spin_count=result.get("spin_count", 0),
            spin_pattern=result.get("spin_pattern"),
            cognitive_load=result.get("cognitive_load", 0.0),
            strategy=result.get("strategy", "explore"),
            uncertainties=result.get("uncertainties", []),
            assumptions=result.get("assumptions", []),
        )

    async def update_cognitive_state(self, state: CognitiveState) -> None:
        """Update agent cognitive state.

        Args:
            state: New cognitive state.
        """
        await self._request(
            "PUT",
            f"/conversations/{state.conversation_id}/agents/{state.agent_id}/cognitive",
            json={
                "spin_count": state.spin_count,
                "spin_pattern": state.spin_pattern,
                "cognitive_load": state.cognitive_load,
                "strategy": state.strategy,
                "uncertainties": state.uncertainties,
                "assumptions": state.assumptions,
            },
        )

    # ========== Background Tasks ==========

    async def queue_background_task(
        self,
        task_type: str,
        prompt: str,
        priority: int = 5,
    ) -> str:
        """Queue a background task.

        Args:
            task_type: Type of task.
            prompt: Task prompt/instructions.
            priority: Priority (1-10, 1 is highest).

        Returns:
            Task ID.
        """
        result = await self._request(
            "POST",
            "/background/tasks",
            json={
                "task_type": task_type,
                "prompt": prompt,
                "priority": priority,
            },
        )
        return result["task_id"]

    async def get_task_result(self, task_id: str) -> dict[str, Any]:
        """Get background task result.

        Args:
            task_id: Task ID.

        Returns:
            Task result.
        """
        return await self._request("GET", f"/background/tasks/{task_id}")

    # ========== Agents ==========

    async def list_agents(self) -> list[dict[str, Any]]:
        """List available agents.

        Returns:
            List of agent info.
        """
        return await self._request("GET", "/agents/")

    async def get_agent_context(self, agent_id: int) -> dict[str, Any]:
        """Get agent context.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent context entries.
        """
        return await self._request("GET", f"/agents/{agent_id}/context")

    # ========== Cleanup ==========

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        await self.disconnect_conversation()
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
