import typer
import asyncio
import re
import shlex
from rich.console import Console
from rich.markdown import Markdown
from typing import Optional, List, Tuple

from config.loader import load_config
from agents.core.coordinator import AgentCoordinator
from models.agent import AgentRole
from core.tooling import ToolCommand

chat_app = typer.Typer(
    name="chat",
    help="Interactive chat session with the agent coordinator.",
)
console = Console()


async def confirm_tool_execution(tool: ToolCommand) -> bool:
    """Ask user for confirmation before executing a tool."""
    console.print(f"\n[bold yellow]Agent wants to execute:[/bold yellow]")
    console.print(f"  [bold]{tool.name}[/bold]: {' '.join(tool.command)}")
    console.print(f"  [dim]{tool.description}[/dim]")

    # Use a loop to get valid input, running in a thread to avoid blocking loop
    response = await asyncio.to_thread(
        console.input, "[bold yellow]Allow execution? (y/N): [/bold yellow]"
    )
    return response.strip().lower() == "y"


def parse_tool_calls(text: str) -> List[Tuple[str, List[str]]]:
    """Parse tool calls from agent output."""
    pattern = re.compile(r"<execute>\s*(.*?)\s*</execute>", re.DOTALL)
    matches = pattern.findall(text)
    calls = []
    for match in matches:
        # Split into command and args, handling quotes
        try:
            parts = shlex.split(match.strip())
            if parts:
                calls.append((parts[0], parts[1:]))
        except ValueError:
            pass  # Ignore malformed
    return calls


@chat_app.callback(invoke_without_command=True)
def chat_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        asyncio.run(start_chat())


async def start_chat():
    """Start an interactive chat session."""
    config = load_config()
    coordinator = AgentCoordinator(config, tool_confirmation_callback=confirm_tool_execution)

    # Setup history/session if needed (similar to orchestration_entrypoint)

    try:
        from core.orchestration_entrypoint import _attach_history

        await _attach_history(coordinator, config)
    except ImportError:
        pass

    # Register default agents from config, or fall back to a generalist.
    if not coordinator.list_agents():
        default_agents = []
        if hasattr(config, "orchestrator"):
            default_agents = list(getattr(config.orchestrator, "default_agents", []) or [])

        if default_agents:
            for agent_cfg in default_agents:
                try:
                    role = AgentRole(agent_cfg.role)
                except ValueError:
                    continue

                await coordinator.register_agent(
                    name=agent_cfg.name,
                    role=role,
                    backend_name=agent_cfg.backend or None,
                    system_prompt=agent_cfg.system_prompt or "",
                )
        else:
            await coordinator.register_agent(
                name="Generalist",
                role=AgentRole.GENERAL,
                backend_name="gemini",
            )

    await coordinator.start_all_agents()

    console.print("[bold green]HAFS Interactive Shell[/bold green]")
    console.print("Type 'exit' or 'quit' to end the session.\n")

    try:
        while True:
            user_input = console.input("[bold blue]You > [/bold blue]")
            if user_input.strip().lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            try:
                # Route message to appropriate agent
                recipient = await coordinator.route_message(user_input, sender="user")

                console.print(f"[dim]Routed to @{recipient}[/dim]")
                console.print(f"[bold green]@{recipient} > [/bold green]", end="")

                # Stream response
                response_text = ""
                async for chunk in coordinator.stream_agent_response(recipient):
                    console.print(chunk, end="")
                    response_text += chunk
                console.print()  # Newline after stream

                # Optional: Render markdown for the final response if it looks like markdown
                # console.print(Markdown(response_text))

                # Handle tool calls
                tool_calls = parse_tool_calls(response_text)
                if tool_calls:
                    for tool_name, args in tool_calls:
                        console.print(
                            f"\n[bold blue]Executing tool:[/bold blue] {tool_name} {args}"
                        )
                        lane = coordinator.get_lane(recipient)
                        if lane:
                            output = await lane.execute_tool(tool_name, args)

                            result_msg = f"Tool '{tool_name}' output:\n{output}"
                            console.print(f"[dim]{result_msg}[/dim]")

                            # Feed back to agent
                            # We inject context so the agent sees the result
                            await lane.inject_context(f"System: {result_msg}")
                            console.print("[dim]Result sent to agent context.[/dim]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")
    finally:
        await coordinator.stop_all_agents()
        console.print("[bold green]Session ended.[/bold green]")


if __name__ == "__main__":
    asyncio.run(start_chat())
