"""Message routing for multi-agent orchestration."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from hafs.agents.roles import match_role_by_keywords

if TYPE_CHECKING:
    from hafs.models.agent import Agent


class MentionRouter:
    """Routes messages to agents based on mentions and content analysis.

    Supports @mention syntax for explicit routing and content-based
    inference for implicit routing.

    Example:
        router = MentionRouter()
        mentions = router.extract_mentions("@planner create a roadmap")
        # mentions = ["planner"]

        recipient = router.route_by_content("implement the login feature", agents)
        # recipient = "coder" (if a coder agent exists)
    """

    # Pattern for extracting @mentions from messages
    MENTION_PATTERN = r"@(\w+)"

    def extract_mentions(self, message: str) -> list[str]:
        """Extract @mentions from a message.

        Args:
            message: The message text to parse.

        Returns:
            List of mentioned agent names (without @ symbol).

        Example:
            extract_mentions("@planner and @coder please collaborate")
            # Returns: ["planner", "coder"]
        """
        matches = re.findall(self.MENTION_PATTERN, message)
        return matches

    def strip_mentions(self, message: str) -> str:
        """Remove @mentions from a message.

        Args:
            message: The message text to clean.

        Returns:
            Message with @mentions removed.

        Example:
            strip_mentions("@planner create a roadmap")
            # Returns: "create a roadmap"
        """
        return re.sub(self.MENTION_PATTERN, "", message).strip()

    def route_by_content(
        self, message: str, agents: dict[str, Agent]
    ) -> str | None:
        """Infer the best agent to handle a message based on content.

        Uses keyword matching to determine which agent role is most
        appropriate for the message content.

        Args:
            message: The message to analyze.
            agents: Dictionary of available agents keyed by name.

        Returns:
            Name of the best agent to handle this message, or None if
            no suitable agent is found.

        Example:
            agents = {"alice": Agent(name="alice", role=AgentRole.CODER, ...)}
            route_by_content("implement a login function", agents)
            # Returns: "alice"
        """
        # Try to match a role based on keywords in the message
        matched_role = match_role_by_keywords(message)
        if not matched_role:
            return None

        # Find the first agent with the matched role
        for agent in agents.values():
            if agent.role == matched_role:
                return agent.name

        return None

    def resolve_recipient(
        self, message: str, agents: dict[str, Agent]
    ) -> tuple[str | None, str]:
        """Resolve the recipient and clean message.

        First tries explicit @mentions, then falls back to content-based
        routing if no mention is found.

        Args:
            message: The message to route.
            agents: Dictionary of available agents.

        Returns:
            Tuple of (recipient_name, cleaned_message). Recipient may be
            None if no suitable agent is found.

        Example:
            resolve_recipient("@coder fix the bug", agents)
            # Returns: ("coder", "fix the bug")

            resolve_recipient("implement login", agents)
            # Returns: ("alice", "implement login") if alice is a coder
        """
        mentions = self.extract_mentions(message)
        cleaned_message = self.strip_mentions(message)

        # If there's an explicit mention, use that
        if mentions:
            # Take the first mention as the recipient
            mentioned_name = mentions[0]
            # Verify the agent exists
            if mentioned_name in agents:
                return mentioned_name, cleaned_message
            # If not found by exact name, try to match by role
            for agent in agents.values():
                if agent.role.value == mentioned_name:
                    return agent.name, cleaned_message

        # Fall back to content-based routing
        recipient = self.route_by_content(message, agents)
        return recipient, cleaned_message

    def has_mentions(self, message: str) -> bool:
        """Check if a message contains any @mentions.

        Args:
            message: The message to check.

        Returns:
            True if the message contains at least one @mention.
        """
        return bool(re.search(self.MENTION_PATTERN, message))

    def is_broadcast(self, message: str, agents: dict[str, Agent]) -> bool:
        """Check if a message should be broadcast to all agents.

        Args:
            message: The message to check.
            agents: Dictionary of available agents.

        Returns:
            True if the message has no explicit mentions and cannot be
            routed to a specific agent.
        """
        recipient, _ = self.resolve_recipient(message, agents)
        return recipient is None
