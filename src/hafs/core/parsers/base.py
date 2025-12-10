"""Abstract base class for log parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseParser(ABC, Generic[T]):
    """Abstract base class for log parsers."""

    def __init__(self, base_path: Path | None = None):
        """Initialize parser with optional custom base path.

        Args:
            base_path: Custom path to search for logs. If None, uses default_path().
        """
        self.base_path = base_path or self.default_path()

    @abstractmethod
    def default_path(self) -> Path:
        """Return the default path for this parser.

        Returns:
            Default Path where this parser looks for data.
        """
        pass

    @abstractmethod
    def parse(self, max_items: int = 50) -> list[T]:
        """Parse and return items.

        Args:
            max_items: Maximum number of items to return.

        Returns:
            List of parsed items.
        """
        pass

    @abstractmethod
    def search(self, query: str, items: list[T] | None = None) -> list[T]:
        """Search items by keyword.

        Args:
            query: Search query string.
            items: Optional pre-parsed items to search. If None, calls parse().

        Returns:
            List of items matching the query.
        """
        pass

    def exists(self) -> bool:
        """Check if the base path exists.

        Returns:
            True if base_path exists.
        """
        return self.base_path.exists()
