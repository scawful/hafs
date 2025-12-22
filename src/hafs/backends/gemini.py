"""Gemini CLI backend using PTY subprocess."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Callable

from backends.base import BackendCapabilities, BaseChatBackend
from backends.pty import PtyOptions, PtyWrapper


def strip_ansi(text: str) -> str:
    """Strip all ANSI escape sequences from text.

    Handles all common escape sequences including:
    - SGR (colors, styles): ESC[...m
    - Cursor movement: ESC[...H, ESC[...A, etc.
    - Erase: ESC[...J, ESC[...K
    - 24-bit color: ESC[38;2;R;G;Bm, ESC[48;2;R;G;Bm
    - OSC (title): ESC]...BEL
    - Partial/malformed sequences

    Args:
        text: Text potentially containing ANSI escape sequences.

    Returns:
        Text with all escape sequences removed.
    """
    # First pass: remove complete escape sequences
    # This pattern matches most ANSI sequences
    ansi_pattern = re.compile(
        r'\x1b'  # ESC character
        r'(?:'
        r'\[[0-9;]*[a-zA-Z]'  # CSI sequences: ESC[...X
        r'|\][^\x07]*\x07'  # OSC sequences: ESC]...BEL
        r'|\][^\x1b]*\x1b\\'  # OSC with ST: ESC]...ESC\
        r'|[PX^_][^\x1b]*\x1b\\'  # DCS/SOS/PM/APC
        r'|[@-Z\\-_]'  # Fe sequences
        r')'
    )
    text = ansi_pattern.sub('', text)

    # Second pass: remove any remaining escape characters and
    # orphaned CSI sequences (e.g., "[38;2;165;153;233m" or "[38;2;165;153;")
    # These can appear when escape sequences are split across chunks
    # Match both complete (ends with letter) and incomplete (ends with digit/semicolon)
    orphan_csi = re.compile(r'\[[\d;]*[a-zA-Z]?')
    text = orphan_csi.sub('', text)

    # Handle tail fragments of escape sequences (e.g., "233m" from a split sequence)
    # This matches digits followed by a single letter at the start of text
    tail_fragment = re.compile(r'^[\d;]+[a-zA-Z]')
    text = tail_fragment.sub('', text)

    # Remove standalone ESC characters
    text = text.replace('\x1b', '')

    # Remove BEL characters
    text = text.replace('\x07', '')

    return text


class GeminiResponseParser:
    """Parser for Gemini CLI output format."""

    # Patterns to filter out CLI chrome
    PROMPT_PATTERN = re.compile(r"^(❯|>|\$)\s*", re.MULTILINE)
    SPINNER_PATTERN = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")
    THINKING_PATTERN = re.compile(r"^Thinking\.\.\.", re.MULTILINE)

    # Partial escape sequence at end of chunk
    PARTIAL_ESCAPE = re.compile(r'\x1b(?:\[[0-9;]*)?$')

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
        # Prepend any buffered partial escape sequence
        if self._buffer:
            chunk = self._buffer + chunk
            self._buffer = ""

        # Check for partial escape sequence at end
        partial_match = self.PARTIAL_ESCAPE.search(chunk)
        if partial_match:
            self._buffer = partial_match.group(0)
            chunk = chunk[:partial_match.start()]

        # Strip all ANSI escape codes (comprehensive)
        text = strip_ansi(chunk)

        # Skip spinner characters
        text = self.SPINNER_PATTERN.sub("", text)

        # Skip thinking indicator
        text = self.THINKING_PATTERN.sub("", text)

        # Skip prompt
        text = self.PROMPT_PATTERN.sub("", text)

        # Clean up whitespace but preserve newlines for formatting
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(line for line in cleaned_lines if line)

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
        self._raw_output_callback: Callable[[str], None] | None = None

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
            # Hook raw output callback if set
            if self._raw_output_callback:
                self._pty.set_output_callback(self._raw_output_callback)

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

    async def stream_response(self) -> AsyncGenerator[str, None]:
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

    def send_key(self, key: str) -> None:
        """Send a special key to the Gemini CLI process.

        Args:
            key: Key name (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        if self._pty:
            self._pty.send_key(key)

    def write_raw(self, data: str) -> None:
        """Write raw data directly to the Gemini CLI PTY stdin.

        Args:
            data: Raw string data to write.
        """
        if self._pty and self._pty.is_running:
            try:
                import os
                if self._pty._master_fd is not None:
                    os.write(self._pty._master_fd, data.encode("utf-8"))
            except OSError:
                pass

    def interrupt(self) -> None:
        """Send Ctrl+C (interrupt) to the Gemini CLI process."""
        if self._pty:
            self._pty.send_interrupt()

    def set_raw_output_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Set callback for raw PTY output (before parsing).

        This allows widgets to receive unprocessed terminal data for
        proper terminal emulation using pyte.

        Args:
            callback: Function called with raw output chunks, or None to clear.
        """
        self._raw_output_callback = callback
        if self._pty:
            self._pty.set_output_callback(callback)

