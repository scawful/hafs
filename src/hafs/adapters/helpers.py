"""Helpers for working with adapter interfaces."""

from __future__ import annotations

from typing import Any, Optional


async def search_issues(adapter: Any, query: str, limit: int = 50) -> list[Any]:
    """Search issues with compatibility fallbacks."""
    if not adapter:
        return []
    if hasattr(adapter, "search_issues"):
        try:
            return await adapter.search_issues(query, limit=limit)
        except TypeError:
            return await adapter.search_issues(query)
    if hasattr(adapter, "search_bugs"):
        return await adapter.search_bugs(query)
    return []


async def get_reviews(adapter: Any, user: Optional[str] = None) -> list[Any]:
    """Get reviews with compatibility fallbacks."""
    if not adapter:
        return []
    if hasattr(adapter, "get_reviews"):
        try:
            return await adapter.get_reviews(user)
        except TypeError:
            return await adapter.get_reviews()
    return []


async def get_submitted_reviews(
    adapter: Any, user: str, limit: int = 5
) -> list[Any]:
    """Get submitted reviews with compatibility fallbacks."""
    if not adapter:
        return []
    if hasattr(adapter, "get_submitted"):
        try:
            return await adapter.get_submitted(user, limit=limit)
        except TypeError:
            return await adapter.get_submitted(user)
    return []


async def search_code(adapter: Any, query: str, limit: int = 10) -> list[Any]:
    """Search code with compatibility fallbacks."""
    if not adapter:
        return []
    if hasattr(adapter, "search"):
        try:
            return await adapter.search(query, limit=limit)
        except TypeError:
            return await adapter.search(query)
    return []


async def read_code_file(adapter: Any, path: str) -> str:
    """Read code file with compatibility fallbacks."""
    if not adapter:
        return ""
    if hasattr(adapter, "read_file"):
        return await adapter.read_file(path)
    return ""
