"""Gemini one-shot CLI backend for fast headless answers."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from backends.base import BackendCapabilities, BaseChatBackend


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

        try:
            if proc.stdout:
                async for chunk in proc.stdout:
                    decoded = chunk.decode("utf-8", errors="replace")
                    if decoded:
                        yield decoded
            await proc.wait()
        finally:
            self._busy = False

    async def inject_context(self, context: str) -> None:
        self._pending_context = context

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_busy(self) -> bool:
        return self._busy


class GeminiOneShotBackend(_OneShotCliBackend):
    @property
    def name(self) -> str:
        return "gemini_oneshot"

    @property
    def display_name(self) -> str:
        return "Gemini One-shot"

    def __init__(
        self,
        project_dir: Path | None = None,
        command: str = "gemini",
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            command=command,
            prompt_flag=["-p"],
            project_dir=project_dir,
            extra_args=extra_args,
            env=env,
        )
