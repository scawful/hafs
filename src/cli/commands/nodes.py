import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from tui.console import nodes as ui_nodes

nodes_app = typer.Typer(
    name="nodes",
    help="Manage distributed node registry and health checks",
)
console = Console()

_EMBED_HINTS = ("embed", "embedding")
_PROBE_SUITES = {
    "smoke": [
        {
            "id": "hook_steps",
            "prompt": "List 3 short steps for adding a SNES ASM hook in Oracle of Secrets.",
            "type": "nonempty",
        },
        {
            "id": "rom_validation",
            "prompt": "Summarize in 2 sentences how to validate a new ROM hook without breaking gameplay.",
            "type": "nonempty",
        },
    ],
    "tool-call": [
        {
            "id": "tool_read_file",
            "prompt": (
                "Return ONLY JSON for a tool call. Schema: "
                "{\"tool\":\"read_file\",\"args\":{\"path\":\"...\"}}. "
                "Task: open ~/Code/docs/README.md."
            ),
            "type": "tool",
            "tool": "read_file",
        },
        {
            "id": "tool_search_symbol",
            "prompt": (
                "Return ONLY JSON for a tool call. Schema: "
                "{\"tool\":\"rg\",\"args\":{\"pattern\":\"...\",\"path\":\"...\"}}. "
                "Task: search for Link_Main in ~/Code/Oracle-of-Secrets."
            ),
            "type": "tool",
            "tool": "rg",
        },
        {
            "id": "tool_write_note",
            "prompt": (
                "Return ONLY JSON for a tool call. Schema: "
                "{\"tool\":\"write_file\",\"args\":{\"path\":\"...\",\"content\":\"...\"}}. "
                "Task: write 'probe ok' to ~/Mounts/notes/probe.txt."
            ),
            "type": "tool",
            "tool": "write_file",
        },
    ],
}


def _is_embedding_model(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _EMBED_HINTS)


def _extract_json_payload(text: str) -> Optional[object]:
    if not text:
        return None
    candidates = [text.strip()]

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1])

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append(text[start_arr : end_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _evaluate_tool_call(text: str, expected_tool: str) -> tuple[bool, str]:
    payload = _extract_json_payload(text)
    if payload is None:
        return False, "no JSON payload"

    if isinstance(payload, list):
        payload = payload[0] if payload else None

    if not isinstance(payload, dict):
        return False, "payload not an object"

    tool = payload.get("tool") or payload.get("name")
    args = payload.get("args") or payload.get("arguments")

    if tool != expected_tool:
        return False, f"tool mismatch ({tool})"
    if not isinstance(args, dict):
        return False, "args missing"

    return True, "ok"


async def _resolve_node(
    name: Optional[str],
    *,
    task_type: Optional[str],
    required_model: Optional[str],
    prefer_gpu: bool,
    prefer_local: bool,
):
    from core.nodes import node_manager

    if name:
        node = node_manager.get_node(name)
        if not node:
            return None
        await node_manager.health_check(node)
        return node

    return await node_manager.get_best_node(
        task_type=task_type,
        required_model=required_model,
        prefer_gpu=prefer_gpu,
        prefer_local=prefer_local,
    )


@nodes_app.callback(invoke_without_command=True)
def nodes_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@nodes_app.command("list")
def list_nodes() -> None:
    """List configured nodes."""
    from core.nodes import node_manager

    async def _list() -> None:
        try:
            await node_manager.load_config()
            ui_nodes.render_nodes_list(console, node_manager.nodes)
        finally:
            await node_manager.close()

    asyncio.run(_list())


@nodes_app.command("status")
def status() -> None:
    """Check node health status."""
    from core.nodes import node_manager

    async def _status() -> None:
        try:
            await node_manager.load_config()
            await node_manager.health_check_all()
            ui_nodes.render_nodes_status(console, node_manager.summary())
        finally:
            await node_manager.close()

    asyncio.run(_status())


@nodes_app.command("show")
def show(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name"),
) -> None:
    """Show detailed node configuration."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from core.nodes import node_manager

    async def _show() -> None:
        try:
            await node_manager.load_config()
            node = node_manager.get_node(name)
            if not node:
                ui_nodes.render_unknown_node(console, name)
                raise typer.Exit(1)
            ui_nodes.render_node_details(console, node.to_dict())
        finally:
            await node_manager.close()

    asyncio.run(_show())


@nodes_app.command("discover")
def discover() -> None:
    """Discover Ollama nodes on Tailscale."""
    from core.nodes import node_manager

    async def _discover() -> None:
        try:
            await node_manager.load_config()
            found = await node_manager.discover_tailscale_nodes()
            ui_nodes.render_discovered_nodes(console, found)
        finally:
            await node_manager.close()

    asyncio.run(_discover())


@nodes_app.command("models")
def models(
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    details: bool = typer.Option(False, "--details", help="Show model metadata"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """List models available on a node."""
    from core.nodes import node_manager

    async def _models() -> None:
        backend = None
        try:
            await node_manager.load_config()
            node = await _resolve_node(
                name,
                task_type=None,
                required_model=None,
                prefer_gpu=prefer_gpu,
                prefer_local=prefer_local,
            )
            if not node:
                ui_nodes.render_unknown_node(console, name or "best")
                raise typer.Exit(1)

            backend = node_manager.create_backend(node)
            if not await backend.start():
                console.print(f"[red]Failed to connect to {node.name}[/red]")
                raise typer.Exit(1)

            model_list = await backend.list_models()
            if details:
                for model in model_list:
                    model_name = model.get("name")
                    if not model_name:
                        continue
                    try:
                        model["show"] = await backend.show_model(model_name)
                    except Exception as exc:
                        model["show"] = {"error": str(exc)}

            ui_nodes.render_models(console, model_list, details)
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

    asyncio.run(_models())


@nodes_app.command("pull")
def pull(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    model: Optional[str] = typer.Argument(None, help="Model name to pull"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """Pull a model onto a node's Ollama instance."""
    if model is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from core.nodes import node_manager

    async def _pull() -> None:
        backend = None
        try:
            await node_manager.load_config()
            node = await _resolve_node(
                name,
                task_type=None,
                required_model=None,
                prefer_gpu=prefer_gpu,
                prefer_local=prefer_local,
            )
            if not node:
                ui_nodes.render_unknown_node(console, name or "best")
                raise typer.Exit(1)

            backend = node_manager.create_backend(node, model=model)
            if not await backend.start():
                console.print(f"[red]Failed to connect to {node.name}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold]Pulling[/bold] {model} on {node.name}...")
            success = await backend.pull_model(
                model,
                progress_callback=lambda status: console.print(f"[dim]{status}[/dim]"),
            )
            ui_nodes.render_pull_result(console, model, success, node.name)
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

    asyncio.run(_pull())


@nodes_app.command("chat")
def chat(
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """Interactive chat session with a node's Ollama model."""
    from core.nodes import node_manager

    async def _chat() -> None:
        backend = None
        try:
            await node_manager.load_config()
            node = await _resolve_node(
                name,
                task_type=None,
                required_model=model,
                prefer_gpu=prefer_gpu,
                prefer_local=prefer_local,
            )
            if not node:
                ui_nodes.render_unknown_node(console, name or "best")
                raise typer.Exit(1)

            backend = node_manager.create_backend(node, model=model)
            if not await backend.start():
                console.print(f"[red]Failed to connect to {node.name}[/red]")
                raise typer.Exit(1)

            if system:
                if hasattr(backend, "add_system_message"):
                    backend.add_system_message(system)
                else:
                    await backend.inject_context(system)

            console.print(
                f"[bold green]HAFS Node Chat[/bold green] ({node.name} / {backend.model})"
            )
            console.print("Type 'exit' to quit, '/reset' to clear history.\n")

            while True:
                user_input = console.input("[bold blue]You > [/bold blue]").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue
                if user_input == "/reset":
                    backend.clear_history()
                    console.print("[dim]History cleared.[/dim]")
                    continue

                await backend.send_message(user_input)
                console.print(f"[bold green]@{backend.model} > [/bold green]", end="")
                async for chunk in backend.stream_response():
                    console.print(chunk, end="")
                console.print()
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

    asyncio.run(_chat())


@nodes_app.command("probe")
def probe(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    prompt: str = typer.Option(
        "Say hello from HAFS.", "--prompt", "-p", help="Prompt to send"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    task_type: Optional[str] = typer.Option(None, "--task", help="Preferred task type"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """Run a one-shot prompt against a node for smoke testing."""
    from core.nodes import node_manager

    async def _probe() -> None:
        backend = None
        try:
            await node_manager.load_config()

            node = await _resolve_node(
                name,
                task_type=task_type,
                required_model=model,
                prefer_gpu=prefer_gpu,
                prefer_local=prefer_local,
            )

            if not node:
                console.print("[red]No suitable node available[/red]")
                raise typer.Exit(1)

            if model and node.models and model not in node.models:
                console.print(
                    f"[yellow]Model '{model}' not reported by {node.name}; "
                    "attempting anyway.[/yellow]"
                )

            backend = node_manager.create_backend(node, model=model)
            if not await backend.start():
                console.print(f"[red]Failed to connect to {node.name}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold]Node:[/bold] {node.name} ({node.host}:{node.port})")
            console.print(f"[bold]Model:[/bold] {backend.model}")
            console.print(f"[bold]Prompt:[/bold] {prompt}")
            response = await backend.generate_one_shot(prompt, system=system)
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

    if prompt is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    asyncio.run(_probe())


@nodes_app.command("probe-suite")
def probe_suite(
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    suite: str = typer.Option("smoke", "--suite", "-s", help="Probe suite name"),
    model: list[str] = typer.Option([], "--model", "-m", help="Model name (repeatable)"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    include_embeddings: bool = typer.Option(
        False, "--include-embeddings", help="Include embedding-only models"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save JSON report"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """Run a probe suite across one or more models."""
    from core.nodes import node_manager

    async def _probe_suite() -> None:
        backend = None
        results: list[dict[str, object]] = []
        started_at = time.time()
        try:
            await node_manager.load_config()
            node = await _resolve_node(
                name,
                task_type=None,
                required_model=None,
                prefer_gpu=prefer_gpu,
                prefer_local=prefer_local,
            )
            if not node:
                ui_nodes.render_unknown_node(console, name or "best")
                raise typer.Exit(1)

            suites = []
            if suite == "all":
                suites = list(_PROBE_SUITES.keys())
            else:
                suites = [s.strip() for s in suite.split(",") if s.strip()]

            cases = []
            for suite_name in suites:
                suite_cases = _PROBE_SUITES.get(suite_name)
                if not suite_cases:
                    console.print(f"[yellow]Unknown suite: {suite_name}[/yellow]")
                    continue
                cases.extend(suite_cases)

            if not cases:
                console.print("[red]No probe cases selected[/red]")
                raise typer.Exit(1)

            models_to_run = model or []
            if not models_to_run:
                await node_manager.health_check(node)
                models_to_run = list(node.models or [])

            if not include_embeddings:
                models_to_run = [m for m in models_to_run if not _is_embedding_model(m)]

            if not models_to_run:
                console.print("[red]No models available for probe suite[/red]")
                raise typer.Exit(1)

            for model_name in models_to_run:
                backend = node_manager.create_backend(node, model=model_name)
                if not await backend.start():
                    results.append(
                        {
                            "model": model_name,
                            "case_id": "connect",
                            "ok": False,
                            "latency_ms": None,
                            "error": "connection failed",
                        }
                    )
                    await backend.stop()
                    continue

                for case in cases:
                    prompt_text = case["prompt"]
                    case_id = case["id"]
                    started = time.time()
                    try:
                        response = await backend.generate_one_shot(
                            prompt_text,
                            system=system,
                        )
                        latency_ms = int((time.time() - started) * 1000)
                        ok = bool(response.strip())
                        note = "ok"
                        if case.get("type") == "tool":
                            ok, note = _evaluate_tool_call(
                                response, case.get("tool", "")
                            )
                        results.append(
                            {
                                "model": model_name,
                                "case_id": case_id,
                                "ok": ok,
                                "latency_ms": latency_ms,
                                "note": note,
                            }
                        )
                    except Exception as exc:
                        latency_ms = int((time.time() - started) * 1000)
                        results.append(
                            {
                                "model": model_name,
                                "case_id": case_id,
                                "ok": False,
                                "latency_ms": latency_ms,
                                "error": str(exc),
                            }
                        )

                await backend.stop()
                backend = None
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

        ui_nodes.render_probe_suite(console, results)

        if output:
            report = {
                "suite": suite,
                "node": name or "best",
                "started_at": started_at,
                "ended_at": time.time(),
                "results": results,
            }
            output_path = output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
            console.print(f"[green]Saved report to {output_path}[/green]")

    asyncio.run(_probe_suite())
