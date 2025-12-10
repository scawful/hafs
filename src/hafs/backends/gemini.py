"""Gemini CLI backend using PTY subprocess."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from hafs.backends.base import BackendCapabilities, BaseChatBackend
from hafs.backends.pty import PtyOptions, PtyWrapper


class GeminiResponseParser:
    """Parser for Gemini CLI output format."""

    # Patterns to filter out CLI chrome
    PROMPT_PATTERN = re.compile(r"^(❯|>|\$)\s*", re.MULTILINE)
    SPINNER_PATTERN = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")
    ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    THINKING_PATTERN = re.compile(r"^Thinking\.\.\.", re.MULTILINE)

    def __init__(self) -> None:
        self._buffer = ""
        self._in_response = False

    def parse_chunk(self, chunk: str) -> str | None:
        """Parse a chunk of Gemini CLI output.

        Args:
            chunk: Raw output chunk from PTY.

        Returns:
            Parsed response text, or None if not part of response.
        """
        # Strip ANSI escape codes
        text = self.ESCAPE_PATTERN.sub("", chunk)

        # Skip spinner characters
        text = self.SPINNER_PATTERN.sub("", text)

        # Skip thinking indicator
        text = self.THINKING_PATTERN.sub("", text)

        # Skip prompt
        text = self.PROMPT_PATTERN.sub("", text)

        # Clean up whitespace
        text = text.strip()

        if text:
            return text
        return None

    def reset(self) -> None:
        """Reset parser state."""
        self._buffer = ""
        self._in_response = False


class GeminiCliBackend(BaseChatBackend):
    """Gemini CLI backend using PTY subprocess.

    Spawns the `gemini` CLI tool and manages interaction via PTY.
    This is the primary backend for work environments.

    Example:
        backend = GeminiCliBackend(project_dir=Path.cwd())
        await backend.start()
        await backend.send_message("Write a Python function")
        async for chunk in backend.stream_response():
            print(chunk, end="")
        await backend.stop()
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        command: str = "gemini",
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        model: str | None = None,
        sandbox: bool = False,
        yolo: bool = False,
        resume: str | None = None,
    ):
        """Initialize Gemini CLI backend.

        Args:
            project_dir: Working directory for gemini CLI.
            command: Command to run (default: "gemini").
            extra_args: Additional CLI arguments.
            env: Environment variables to set.
            model: Model to use (-m flag).
            sandbox: Enable sandbox (-s flag).
            yolo: Enable YOLO mode (-y flag).
            resume: Resume session (-r flag).
        """
        self._project_dir = project_dir
        self._command = command
        self._extra_args = extra_args or []
        self._env = env or {}
        
        # Add flags based on args
        if model:
            self._extra_args.extend(["-m", model])
        if sandbox:
            self._extra_args.append("-s")
        if yolo:
            self._extra_args.append("-y")
        if resume:
            self._extra_args.extend(["-r", resume])

        self._pty: PtyWrapper | None = None
        self._parser = GeminiResponseParser()
        self._busy = False
        self._pending_context: str | None = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini CLI"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_images=False,
            supports_files=True,
            max_context_tokens=1000000,  # Gemini has large context
        )

    async def start(self) -> bool:
        """Start the Gemini CLI process.

        Returns:
            True if started successfully.
        """
        if self._pty and self._pty.is_running:
            return True

        cmd = [self._command] + self._extra_args

        options = PtyOptions(
            env=self._env,
            working_dir=str(self._project_dir) if self._project_dir else None,
        )

        self._pty = PtyWrapper(cmd, options)

        try:
            success = await self._pty.start()
            if success:
                self._parser.reset()
                # Wait a brief moment for the prompt to appear
                # This helps prevent sending input before the shell is ready
                import asyncio
                await asyncio.sleep(0.5)
            return success
        except RuntimeError:
            return False

    async def stop(self) -> None:
        """Stop the Gemini CLI process."""
        if self._pty:
            await self._pty.terminate()
            self._pty = None
            self._parser.reset()

    async def send_message(self, message: str) -> None:
        """Send a message to Gemini CLI.

        Args:
            message: The message to send.

        Raises:
            RuntimeError: If backend is not running.
        """
        if not self._pty or not self._pty.is_running:
            raise RuntimeError("Gemini backend not running")

        self._busy = True
        self._parser.reset()

        # Prepend context if pending
        if self._pending_context:
            full_message = f"{self._pending_context}\n\n{message}"
            self._pending_context = None
        else:
            full_message = message

        # Send message with newline
        await self._pty.write(full_message + "\n")

    async def stream_response(self) -> AsyncIterator[str]:
        """Stream response chunks from Gemini CLI.

        Yields:
            Parsed response text chunks.
        """
        if not self._pty:
            return

        try:
            async for chunk in self._pty.read_output():
                parsed = self._parser.parse_chunk(chunk)
                if parsed:
                    yield parsed
        finally:
            self._busy = False

    async def inject_context(self, context: str) -> None:
        """Inject context to prepend to next message.

        Args:
            context: Context text to inject.
        """
        self._pending_context = context

    @property
    def is_running(self) -> bool:
        return self._pty is not None and self._pty.is_running

    @property
    def is_busy(self) -> bool:
        return self._busy


# Auto-register on import
from hafs.backends.base import BackendRegistry

BackendRegistry.register(GeminiCliBackend)
