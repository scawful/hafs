"""Parser registry with plugin support."""

from __future__ import annotations

from typing import Type

from hafs.core.parsers.base import BaseParser


class ParserRegistry:
    """Registry for log parsers with plugin support.

    Allows registration and lookup of parser classes by name.
    Supports both built-in and custom parsers.
    """

    _parsers: dict[str, Type[BaseParser]] = {}  # type: ignore[type-arg]

    @classmethod
    def register(cls, name: str, parser_class: Type[BaseParser]) -> None:  # type: ignore[type-arg]
        """Register a parser class.

        Args:
            name: Unique name for the parser.
            parser_class: Parser class to register.
        """
        cls._parsers[name] = parser_class

    @classmethod
    def get(cls, name: str) -> Type[BaseParser] | None:  # type: ignore[type-arg]
        """Get a parser class by name.

        Args:
            name: Name of the parser.

        Returns:
            Parser class or None if not found.
        """
        return cls._parsers.get(name)

    @classmethod
    def list_parsers(cls) -> list[str]:
        """List all registered parser names.

        Returns:
            List of registered parser names.
        """
        return list(cls._parsers.keys())

    @classmethod
    def load_defaults(cls) -> None:
        """Load built-in parsers."""
        from hafs.core.parsers.gemini import GeminiLogParser
        from hafs.core.parsers.claude import ClaudePlanParser
        from hafs.core.parsers.antigravity import AntigravityParser

        cls.register("gemini", GeminiLogParser)
        cls.register("claude", ClaudePlanParser)
        cls.register("antigravity", AntigravityParser)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered parsers (useful for testing)."""
        cls._parsers.clear()


# Auto-load defaults on import
ParserRegistry.load_defaults()
