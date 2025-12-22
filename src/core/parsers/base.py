"""Abstract base class for log parsers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Generic, TypeVar

from core.search import SearchResult, fuzzy_filter_multi

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseParser(ABC, Generic[T]):
    """Abstract base class for log parsers."""

    def __init__(self, base_path: Path | None = None):
        """Initialize parser with optional custom base path.

        Args:
            base_path: Custom path to search for logs. If None, uses default_path().
        """
        self.base_path = base_path or self.default_path()
        self._last_error: str | None = None

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

    def fuzzy_search(
        self,
        query: str,
        items: list[T] | None = None,
        keys: dict[str, Callable[[T], str]] | None = None,
        threshold: float = 60,
    ) -> list[SearchResult[T]]:
        """Fuzzy search items.

        Args:
            query: Search query string.
            items: Optional pre-parsed items to search. If None, calls parse().
            keys: Dictionary mapping field names to extraction functions.
            threshold: Minimum fuzzy score (0-100).

        Returns:
            List of SearchResult with items and scores.
        """
        if items is None:
            items = self.parse()

        if keys is None:
            keys = self._get_search_keys()

        return fuzzy_filter_multi(query, items, keys, threshold)

    def _get_search_keys(self) -> dict[str, Callable[[T], str]]:
        """Get searchable field extractors for this parser type.

        Override in subclasses to define searchable fields.

        Returns:
            Dictionary of field_name -> extraction function.
        """
        return {}

    def exists(self) -> bool:
        """Check if the base path exists.

        Returns:
            True if base_path exists.
        """
        return self.base_path.exists()

    @property
    def last_error(self) -> str | None:
        """Get the last error message from parsing."""
        return self._last_error

    def _set_error(self, error: str) -> None:
        """Set error message for debugging."""
        self._last_error = error
        logger.warning(f"{self.__class__.__name__}: {error}")

    def delete_item(self, item: T) -> bool:
        """Delete an item (session/log).

        Override in subclasses to implement deletion.

        Args:
            item: The item to delete.

        Returns:
            True if deletion was successful.
        """
        return False

    def save_to_context(self, item: T, context_dir: Path) -> Path | None:
        """Save an item to a context directory for permanent storage.

        Override in subclasses to implement saving.

        Args:
            item: The item to save.
            context_dir: Directory to save the context file to.

        Returns:
            Path to the saved file, or None if failed.
        """
        return None

    def get_item_path(self, item: T) -> Path | None:
        """Get the file path for an item.

        Override in subclasses to return the source file path.

        Args:
            item: The item to get the path for.

        Returns:
            Path to the item's source file, or None if not applicable.
        """
        return None
