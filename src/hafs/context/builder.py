"""Context prompt builder for enriching AI agent prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hafs.core.afs.manager import AFSManager
    from hafs.core.parsers.registry import ParserRegistry
    from hafs.models.afs import ContextRoot
    from hafs.models.agent import SharedContext


class ContextPromptBuilder:
    """Builds context-enriched prompts for AI agents.

    Combines user messages with:
    - Scratchpad contents (active reasoning/plans)
    - Memory file listings
    - Recent AI log summaries (Claude plans, Gemini sessions)

    Example:
        builder = ContextPromptBuilder(afs_manager, parser_registry)
        prompt = builder.build(
            "Write a function",
            context_root=my_context,
            include_scratchpad=True,
        )
    """

    def __init__(
        self,
        afs_manager: "AFSManager | None" = None,
        parsers: "ParserRegistry | None" = None,
    ):
        """Initialize context builder.

        Args:
            afs_manager: AFS manager for context access.
            parsers: Parser registry for log parsing.
        """
        self._afs = afs_manager
        self._parsers = parsers

    def build(
        self,
        user_message: str,
        context_root: "ContextRoot | None" = None,
        include_scratchpad: bool = True,
        include_memory: bool = True,
        include_logs: bool = False,
        max_tokens: int = 4000,
        shared_context: "SharedContext | None" = None,
    ) -> str:
        """Build a context-enriched prompt.

        Combines user message with relevant context from:
        - Scratchpad directory contents
        - Memory file listings
        - Recent AI log summaries

        Args:
            user_message: The user's message/query.
            context_root: AFS context root to use.
            include_scratchpad: Include scratchpad contents.
            include_memory: Include memory file listing.
            include_logs: Include recent AI log summaries.
            max_tokens: Approximate token limit for context.
            shared_context: Shared agent context (task, findings, pinned paths).

        Returns:
            Context-enriched prompt string.
        """
        parts: list[str] = []

        if context_root:
            # Scratchpad contents (active reasoning)
            if include_scratchpad:
                scratchpad = self._get_scratchpad_contents(context_root)
                if scratchpad:
                    parts.append("=== ACTIVE SCRATCHPAD ===")
                    parts.append(scratchpad)
                    parts.append("")

            # Memory file listing
            if include_memory:
                memory_listing = self._get_memory_listing(context_root)
                if memory_listing:
                    parts.append("=== AVAILABLE MEMORY FILES ===")
                    parts.append(memory_listing)
                    parts.append("")

        if shared_context and shared_context.context_items:
            parts.append("=== PINNED CONTEXT ITEMS ===")
            for path in shared_context.context_items[:10]:
                parts.append(f"  - {path}")
            if len(shared_context.context_items) > 10:
                parts.append(f"  [and {len(shared_context.context_items) - 10} more...]")
            parts.append("")

        # Recent AI activity logs
        if include_logs and self._parsers:
            log_summary = self._get_log_summary()
            if log_summary:
                parts.append("=== RECENT AI ACTIVITY ===")
                parts.append(log_summary)
                parts.append("")

        # User message last
        parts.append("=== USER REQUEST ===")
        parts.append(user_message)

        full_prompt = "\n".join(parts)

        # Truncate if over token limit (rough estimate: ~4 chars per token)
        char_limit = max_tokens * 4
        if len(full_prompt) > char_limit:
            # Keep the user message intact, truncate context
            context_parts = "\n".join(parts[:-2])
            available = char_limit - len(parts[-1]) - len(parts[-2]) - 10
            if available > 0:
                truncated_context = context_parts[-available:]
                full_prompt = (
                    f"[...truncated...]\n{truncated_context}\n\n"
                    f"=== USER REQUEST ===\n{user_message}"
                )
            else:
                full_prompt = f"=== USER REQUEST ===\n{user_message}"

        return full_prompt

    def _get_scratchpad_contents(self, context_root: "ContextRoot") -> str:
        """Read scratchpad directory contents.

        Args:
            context_root: The context root to read from.

        Returns:
            Formatted scratchpad contents.
        """
        from hafs.models.afs import MountType

        scratchpad_mounts = context_root.get_mounts(MountType.SCRATCHPAD)
        contents: list[str] = []

        for mount in scratchpad_mounts:
            if mount.source.is_file():
                try:
                    text = mount.source.read_text()[:2000]
                    contents.append(f"--- {mount.name} ---\n{text}")
                except OSError:
                    pass
            elif mount.source.is_dir():
                for file in mount.source.glob("*.md"):
                    try:
                        text = file.read_text()[:1000]
                        contents.append(f"--- {file.name} ---\n{text}")
                    except OSError:
                        pass

        return "\n\n".join(contents)

    def _get_memory_listing(self, context_root: "ContextRoot") -> str:
        """List files in memory/knowledge directories.

        Args:
            context_root: The context root to scan.

        Returns:
            Formatted file listing.
        """
        from hafs.models.afs import MountType

        lines: list[str] = []

        for mt in [MountType.MEMORY, MountType.KNOWLEDGE]:
            mounts = context_root.get_mounts(mt)
            for mount in mounts:
                if mount.source.is_dir():
                    for file in mount.source.rglob("*"):
                        if file.is_file():
                            rel_path = file.relative_to(mount.source)
                            lines.append(f"  {mt.value}/{mount.name}/{rel_path}")
                else:
                    lines.append(f"  {mt.value}/{mount.name}")

        return "\n".join(lines)

    def _get_log_summary(self) -> str:
        """Summarize recent AI logs.

        Returns:
            Formatted log summary.
        """
        if not self._parsers:
            return ""

        summaries: list[str] = []

        # Claude plans
        try:
            claude_parser_cls = self._parsers.get("claude")
            if claude_parser_cls:
                claude_parser = claude_parser_cls()
                plans = claude_parser.parse(max_items=3)
                for plan in plans:
                    if hasattr(plan, "title") and hasattr(plan, "progress"):
                        done, total = plan.progress
                        summaries.append(f"Plan: {plan.title} ({done}/{total} tasks)")
        except Exception:
            pass

        # Gemini sessions
        try:
            gemini_parser_cls = self._parsers.get("gemini")
            if gemini_parser_cls:
                gemini_parser = gemini_parser_cls()
                sessions = gemini_parser.parse(max_items=3)
                for session in sessions:
                    if hasattr(session, "short_id") and hasattr(session, "user_message_count"):
                        summaries.append(
                            f"Session: {session.short_id} ({session.user_message_count} messages)"
                        )
        except Exception:
            pass

        return "\n".join(summaries)

    def build_for_agent(
        self,
        user_message: str,
        agent_role: str,
        context_root: "ContextRoot | None" = None,
        shared_context: dict[str, Any] | None = None,
    ) -> str:
        """Build a prompt tailored for a specific agent role.

        Includes role-specific instructions and shared context.

        Args:
            user_message: The user's message/query.
            agent_role: The agent's role (planner, coder, etc).
            context_root: AFS context root to use.
            shared_context: Shared transactive memory context.

        Returns:
            Role-aware context-enriched prompt.
        """
        parts: list[str] = []

        # Shared context (transactive memory)
        if shared_context:
            if shared_context.get("active_task"):
                parts.append(f"Current Task: {shared_context['active_task']}")

            if shared_context.get("findings"):
                parts.append("Recent Findings:")
                for finding in shared_context["findings"][-5:]:
                    parts.append(f"  - {finding}")

            if shared_context.get("decisions"):
                parts.append("Team Decisions:")
                for decision in shared_context["decisions"][-3:]:
                    parts.append(f"  - {decision}")

            if shared_context.get("context_items"):
                parts.append("Pinned Context:")
                for path in shared_context["context_items"][:8]:
                    parts.append(f"  - {path}")

            parts.append("")

        # Role-specific context
        role_context = self._get_role_context(agent_role, context_root)
        if role_context:
            parts.append(role_context)
            parts.append("")

        # User message
        parts.append(user_message)

        return "\n".join(parts)

    def _get_role_context(
        self,
        role: str,
        context_root: "ContextRoot | None",
    ) -> str:
        """Get role-specific context.

        Args:
            role: Agent role name.
            context_root: AFS context root.

        Returns:
            Role-specific context string.
        """
        if not context_root:
            return ""

        role = role.lower()

        if role == "planner":
            # Planners get memory overview
            return self._get_memory_listing(context_root)
        elif role == "coder":
            # Coders get scratchpad (current plans/todos)
            return self._get_scratchpad_contents(context_root)
        elif role == "critic":
            # Critics get both for review
            memory = self._get_memory_listing(context_root)
            scratchpad = self._get_scratchpad_contents(context_root)
            return f"Memory Files:\n{memory}\n\nCurrent Scratchpad:\n{scratchpad}"
        elif role == "researcher":
            # Researchers get knowledge focus
            from hafs.models.afs import MountType

            knowledge_mounts = context_root.get_mounts(MountType.KNOWLEDGE)
            items = []
            for mount in knowledge_mounts:
                items.append(f"  - {mount.name}: {mount.source}")
            return "Available Knowledge:\n" + "\n".join(items)

        return ""
