"""Claude one-shot CLI backend for fast headless answers."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

from backends.base import BackendCapabilities, BaseChatBackend


def check_claude_usage() -> dict:
    """Check Claude CLI usage status.

    Returns:
        Dict with 'available' bool and optional 'message'.
    """
    try:
        # Quick check - just see if claude command exists and responds
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"available": False, "message": "Claude CLI not responding"}
        return {"available": True}
    except FileNotFoundError:
        return {"available": False, "message": "Claude CLI not installed"}
    except subprocess.TimeoutExpired:
        return {"available": False, "message": "Claude CLI timed out"}
    except Exception as e:
        return {"available": False, "message": str(e)}


class _OneShotCliBackend(BaseChatBackend):
    """Run a CLI once per message using subprocess pipes.

    This is meant for headless mode where we don't need interactive terminal
    emulation, and want to avoid PTY startup/overhead.
    """

    def __init__(
        self,
        command: str,
        prompt_flag: list[str],
        project_dir: Path | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._prompt_flag = prompt_flag
        self._project_dir = project_dir
        self._extra_args = extra_args or []
        self._env = env or {}

        self._pending_context: str | None = None
        self._pending_message: str | None = None
        self._busy: bool = False

    @property
    def capabilities(self) -> BackendCapabilities:
        # Tools/files support depends on the CLI; conservative defaults.
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=False,
            supports_images=False,
            supports_files=True,
            max_context_tokens=128000,
        )

    async def start(self) -> bool:
        # No persistent process in one-shot mode.
        return True

    async def stop(self) -> None:
        self._pending_message = None
        self._busy = False

    async def send_message(self, message: str) -> None:
        if self._pending_context:
            message = f"{self._pending_context}\n\n{message}"
            self._pending_context = None
        self._pending_message = message
        self._busy = True

    async def stream_response(self) -> AsyncGenerator[str, None]:
        if not self._pending_message:
            return

        message = self._pending_message
        self._pending_message = None

        cmd = [self._command, *self._extra_args, *self._prompt_flag, message]
        try:
            env = os.environ.copy()
            env.update(self._env)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._project_dir) if self._project_dir else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError:
            yield f"{self._command}: command not found"
            self._busy = False
            return

        # Track if we hit rate limit
        self._hit_rate_limit = False

        try:
            if proc.stdout:
                async for chunk in proc.stdout:
                    decoded = chunk.decode("utf-8", errors="replace")
                    if decoded:
                        # Check for rate limit indicators
                        lower = decoded.lower()
                        if "rate limit" in lower or "usage limit" in lower or "quota" in lower:
                            self._hit_rate_limit = True
                        yield decoded
            await proc.wait()
        finally:
            self._busy = False

    @property
    def hit_rate_limit(self) -> bool:
        """Check if last request hit rate limit."""
        return getattr(self, "_hit_rate_limit", False)

    async def inject_context(self, context: str) -> None:
        self._pending_context = context

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_busy(self) -> bool:
        return self._busy


class ClaudeOneShotBackend(_OneShotCliBackend):
    """Claude CLI backend that avoids creating junk conversations.

    By default:
    - Sessions are NOT persisted (--no-session-persistence)
    - Rate limits are detected and exposed via hit_rate_limit property

    Use skip_if_unavailable=True to silently fail if Claude CLI isn't working.
    """

    @property
    def name(self) -> str:
        return "claude_oneshot"

    @property
    def display_name(self) -> str:
        return "Claude One-shot"

    def __init__(
        self,
        project_dir: Path | None = None,
        command: str = "claude",
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        persist_session: bool = False,
        skip_if_unavailable: bool = False,
    ) -> None:
        """Initialize Claude one-shot backend.

        Args:
            project_dir: Working directory.
            command: CLI command (default: "claude").
            extra_args: Additional CLI args.
            env: Environment variables.
            persist_session: If True, sessions are saved (creates conversations).
            skip_if_unavailable: If True, silently skip if Claude CLI unavailable.
        """
        self._skip_if_unavailable = skip_if_unavailable

        # Check availability if requested
        if skip_if_unavailable:
            status = check_claude_usage()
            if not status.get("available"):
                self._unavailable_reason = status.get("message", "Unknown")
            else:
                self._unavailable_reason = None
        else:
            self._unavailable_reason = None

        # Build args with session persistence control
        args = list(extra_args or [])
        if not persist_session and "--no-session-persistence" not in args:
            args.append("--no-session-persistence")

        super().__init__(
            command=command,
            prompt_flag=["-p"],
            project_dir=project_dir,
            extra_args=args,
            env=env,
        )

    @property
    def is_available(self) -> bool:
        """Check if Claude CLI is available."""
        return self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        """Get reason why Claude CLI is unavailable."""
        return self._unavailable_reason
