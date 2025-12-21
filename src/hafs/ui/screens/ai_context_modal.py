"""AI Context Generation Modal for HAFS TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RichLog, Static

if TYPE_CHECKING:
    from hafs.config.loader import HafsConfig


class AIContextModal(ModalScreen[bool]):
    """Modal for AI-assisted context generation.

    Allows users to prompt an AI agent to create context files
    with proper AFS structure and permissions.
    """

    DEFAULT_CSS = """
    AIContextModal {
        align: center middle;
    }

    AIContextModal #dialog {
        width: 90;
        height: 30;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    AIContextModal #title {
        width: 100%;
        text-align: center;
        padding-bottom: 1;
        color: $primary;
    }

    AIContextModal #prompt-input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    AIContextModal #output-log {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    AIContextModal #button-row {
        height: 3;
        layout: horizontal;
        align: center middle;
    }

    AIContextModal Button {
        margin: 0 1;
    }

    AIContextModal #status {
        height: 1;
        text-align: center;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        target_path: Path,
        config: "HafsConfig | None" = None,
        backend: str = "gemini",
    ) -> None:
        """Initialize AI context modal.

        Args:
            target_path: Directory where context will be created.
            config: HAFS configuration.
            backend: AI backend to use (gemini, claude).
        """
        super().__init__()
        self.target_path = target_path
        self.config = config
        self.backend = backend
        self._task: asyncio.Task | None = None
        self._cancelled = False

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="dialog"):
            yield Label(
                f"[bold]AI Context Generator[/bold]\n"
                f"[dim]Target: {self.target_path}[/dim]",
                id="title",
            )

            yield Input(
                placeholder=(
                    "Describe what you want the AI to create (e.g., "
                    "'Create a memory directory with project goals and design decisions')..."
                ),
                id="prompt-input",
            )

            yield RichLog(id="output-log", highlight=True, markup=True)

            yield Static("[dim]Ready to generate context[/dim]", id="status")

            with Container(id="button-row"):
                yield Button("Generate", variant="primary", id="generate-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        """Focus prompt input on mount."""
        self.query_one("#prompt-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "generate-btn":
            self._start_generation()
        elif event.button.id == "cancel-btn":
            self._cancel_and_close()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter on prompt input."""
        self._start_generation()

    def _start_generation(self) -> None:
        """Start the AI context generation."""
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self._log("[red]Please enter a prompt describing the context you want.[/red]")
            return

        # Disable generate button during execution
        self.query_one("#generate-btn", Button).disabled = True
        self._update_status("Generating context...")

        # Start background task
        self._task = asyncio.create_task(self._generate_context(prompt))

    async def _generate_context(self, user_prompt: str) -> None:
        """Generate context using AI backend.

        Args:
            user_prompt: User's description of desired context.
        """
        self._log(f"[cyan]Starting {self.backend} backend...[/cyan]")

        try:
            # Build the system prompt for context generation
            system_prompt = self._build_context_prompt(user_prompt)

            self._log(f"[dim]Target directory: {self.target_path}[/dim]")
            self._log(f"[dim]Prompt: {user_prompt}[/dim]")
            self._log("")

            # Try to use the specified backend
            if self.backend == "gemini":
                await self._run_gemini(system_prompt)
            elif self.backend == "claude":
                await self._run_claude(system_prompt)
            else:
                self._log(f"[red]Unknown backend: {self.backend}[/red]")
                return

            if not self._cancelled:
                self._log("")
                self._log("[green]Context generation complete![/green]")
                self._update_status("Done! Close to return.")

        except Exception as e:
            self._log(f"[red]Error: {e}[/red]")
            self._update_status("Error occurred")
        finally:
            self.query_one("#generate-btn", Button).disabled = False

    def _build_context_prompt(self, user_prompt: str) -> str:
        """Build the full prompt for context generation.

        Args:
            user_prompt: User's description.

        Returns:
            Full system prompt for the AI.
        """
        return f"""You are helping create AFS (Agentic File System) context files.

Target directory: {self.target_path}

The AFS structure uses these mount types:
- memory/: Project goals, decisions, design rationale (read-only for agents)
- knowledge/: Documentation, guides, reference material (read-only)
- tools/: Scripts, utilities, executables (read-only)
- scratchpad/: Working files, drafts, temp data (read-write)
- history/: Logs, previous versions, audit trail (append-only)

User request: {user_prompt}

Please:
1. Analyze what context files would be helpful
2. Create appropriate markdown files with useful content
3. Use the correct AFS directory structure
4. Include relevant boilerplate and placeholders

Respond with the files you're creating and their content.
"""

    async def _run_gemini(self, prompt: str) -> None:
        """Run Gemini CLI for context generation."""

        self._log("[cyan]Invoking Gemini CLI...[/cyan]")

        try:
            # Run gemini CLI with the prompt
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "-p", prompt,
                cwd=str(self.target_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Stream output
            if process.stdout:
                async for line in process.stdout:
                    if self._cancelled:
                        process.terminate()
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    if decoded:
                        self._log(decoded)

            await process.wait()

            if process.returncode != 0 and not self._cancelled:
                self._log(f"[yellow]Process exited with code {process.returncode}[/yellow]")

        except FileNotFoundError:
            self._log("[red]Gemini CLI not found. Please install it first.[/red]")
            self._log("[dim]Try: pip install google-genai (or google-generativeai for legacy)[/dim]")

    async def _run_claude(self, prompt: str) -> None:
        """Run Claude CLI for context generation."""

        self._log("[cyan]Invoking Claude CLI...[/cyan]")

        try:
            # Run claude CLI with the prompt
            process = await asyncio.create_subprocess_exec(
                "claude",
                "-p", prompt,
                cwd=str(self.target_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Stream output
            if process.stdout:
                async for line in process.stdout:
                    if self._cancelled:
                        process.terminate()
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    if decoded:
                        self._log(decoded)

            await process.wait()

            if process.returncode != 0 and not self._cancelled:
                self._log(f"[yellow]Process exited with code {process.returncode}[/yellow]")

        except FileNotFoundError:
            self._log("[red]Claude CLI not found. Please install it first.[/red]")
            self._log("[dim]See: https://github.com/anthropics/claude-code[/dim]")

    def _log(self, message: str) -> None:
        """Write a message to the output log."""
        log = self.query_one("#output-log", RichLog)
        log.write(message)

    def _update_status(self, status: str) -> None:
        """Update the status text."""
        self.query_one("#status", Static).update(f"[dim]{status}[/dim]")

    def _cancel_and_close(self) -> None:
        """Cancel generation and close modal."""
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()
        self.dismiss(False)

    def on_key(self, event) -> None:
        """Handle escape to close."""
        if event.key == "escape":
            self._cancel_and_close()
