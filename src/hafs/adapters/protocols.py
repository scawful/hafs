"""Adapter protocols for external provider integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class IssueRecord:
    """Normalized issue record for issue tracker adapters."""

    id: str
    title: str
    status: str = ""
    priority: str = ""
    url: str = ""
    assignee: str = ""


@runtime_checkable
class IssueTrackerAdapter(Protocol):
    """Protocol for issue tracker adapters."""

    @property
    def name(self) -> str:
        """Adapter name."""
        ...

    async def connect(self) -> bool:
        """Connect to provider."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from provider."""
        ...

    async def search_issues(self, query: str, limit: int = 50) -> list[IssueRecord]:
        """Search issues by query."""
        ...


@runtime_checkable
class CodeReviewAdapter(Protocol):
    """Protocol for code review adapters."""

    @property
    def name(self) -> str:
        """Adapter name."""
        ...

    async def connect(self) -> bool:
        """Connect to provider."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from provider."""
        ...

    async def get_reviews(self, user: Optional[str] = None) -> list[Any]:
        """Fetch reviews for a user."""
        ...

    async def get_submitted(self, user: str, limit: int = 5) -> list[Any]:
        """Fetch recently submitted reviews for a user."""
        ...


@runtime_checkable
class CodeSearchAdapter(Protocol):
    """Protocol for code search adapters."""

    @property
    def name(self) -> str:
        """Adapter name."""
        ...

    async def connect(self) -> bool:
        """Connect to provider."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from provider."""
        ...

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        """Search code by query."""
        ...

    async def read_file(self, path: str) -> str:
        """Read file contents."""
        ...
