import asyncio
import json
import os
import time
from typing import Optional

import typer
from rich.console import Console

from backends.api.llamacpp import LlamaCppBackend
from hafs.config.loader import load_config

llamacpp_app = typer.Typer(
    name="llamacpp",
    help="Check llama.cpp connection health.",
)
console = Console()


def _resolve_api_key(config) -> Optional[str]:
    if not config:
        return None
    env_name = getattr(config, "api_key_env", None)
    if env_name:
        return os.environ.get(env_name) or None
    return None


def _resolve_settings(
    cfg,
    *,
    base_url: Optional[str],
    host: Optional[str],
    port: Optional[int],
    model: Optional[str],
    timeout: Optional[float],
    max_tokens: Optional[int],
    temperature: Optional[float],
    context_size: Optional[int],
) -> dict[str, object]:
    resolved_base_url = base_url or (cfg.base_url if cfg else None)
    resolved_host = host or (cfg.host if cfg else None)
    resolved_port = port or (cfg.port if cfg else None)
    resolved_model = model or (cfg.model if cfg else None) or "qwen3-14b"
    resolved_timeout = timeout or (cfg.timeout_seconds if cfg else None) or 300.0
    resolved_context = context_size or (cfg.context_size if cfg else None)
    resolved_max_tokens = max_tokens or (cfg.max_tokens if cfg else None) or 4096
    resolved_temperature = (
        temperature if temperature is not None else (cfg.temperature if cfg else None)
    )
    if resolved_temperature is None:
        resolved_temperature = 0.7

    return {
        "base_url": resolved_base_url,
        "host": resolved_host,
        "port": resolved_port,
        "model": resolved_model,
        "timeout": resolved_timeout,
        "context_size": resolved_context,
        "max_tokens": resolved_max_tokens,
        "temperature": resolved_temperature,
    }


@llamacpp_app.command("status")
def status(
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Override base URL (e.g. http://host:11435/v1).",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Override host if base URL is not set.",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="Override port if base URL is not set.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model alias to report against.",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON output.",
    ),
) -> None:
    """Check llama.cpp health and list available models."""

    async def _status() -> None:
        config = load_config()
        cfg = getattr(config, "llamacpp", None)

        settings = _resolve_settings(
            cfg,
            base_url=base_url,
            host=host,
            port=port,
            model=model,
            timeout=timeout,
            max_tokens=None,
            temperature=None,
            context_size=None,
        )
        api_key = _resolve_api_key(cfg)

        backend = LlamaCppBackend(
            base_url=settings["base_url"],
            host=settings["host"],
            port=settings["port"],
            model=settings["model"],
            timeout=settings["timeout"],
            api_key=api_key,
            context_size=settings["context_size"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

        started = time.time()
        try:
            health = await backend.check_health()
            latency_ms = int((time.time() - started) * 1000)
        finally:
            await backend.stop()

        if json_output:
            payload = dict(health)
            payload["latency_ms"] = latency_ms
            console.print(json.dumps(payload, indent=2))
            return

        status_value = health.get("status", "unknown")
        color = "green" if status_value in {"online", "ok"} else "red"
        console.print(f"[bold]{settings['model']}[/bold]")
        console.print(f"[{color}]Status:[/{color}] {status_value} ({latency_ms} ms)")
        base = health.get("base_url") or settings["base_url"] or "auto"
        console.print(f"[bold]Base URL:[/bold] {base}")

        models = health.get("models") or []
        if models:
            console.print(f"[bold]Models:[/bold] {', '.join(models)}")
        else:
            console.print("[bold]Models:[/bold] none reported")

    asyncio.run(_status())


@llamacpp_app.command("probe")
def probe(
    prompt: str = typer.Option(
        "Say hello from HAFS.", "--prompt", "-p", help="Prompt to send"
    ),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    warmup: bool = typer.Option(
        False,
        "--warmup",
        help="Run a small warmup request before the probe.",
    ),
    retries: int = typer.Option(
        0,
        "--retries",
        help="Number of retries if the probe request fails.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Override base URL (e.g. http://host:11435/v1).",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Override host if base URL is not set.",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="Override port if base URL is not set.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model alias to query.",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Override max tokens for the probe request.",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Override temperature for the probe request.",
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON output.",
    ),
) -> None:
    """Run a one-shot prompt against llama.cpp for quick validation."""

    async def _probe() -> None:
        config = load_config()
        cfg = getattr(config, "llamacpp", None)
        settings = _resolve_settings(
            cfg,
            base_url=base_url,
            host=host,
            port=port,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            context_size=None,
        )
        api_key = _resolve_api_key(cfg)

        backend = LlamaCppBackend(
            base_url=settings["base_url"],
            host=settings["host"],
            port=settings["port"],
            model=settings["model"],
            timeout=settings["timeout"],
            api_key=api_key,
            context_size=settings["context_size"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

        started = time.time()
        try:
            if warmup:
                warmup_tokens = max(1, min(8, int(settings["max_tokens"])))
                try:
                    await backend.generate_one_shot(
                        "Warmup.",
                        system=system,
                        max_tokens=warmup_tokens,
                        temperature=0.0,
                    )
                except Exception as exc:
                    if not json_output:
                        console.print(f"[yellow]Warmup failed:[/yellow] {exc}")

            attempts = max(0, retries) + 1
            response = ""
            last_error: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    response = await backend.generate_one_shot(
                        prompt,
                        system=system,
                        max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"],
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= attempts - 1:
                        raise
        finally:
            await backend.stop()

        latency_ms = int((time.time() - started) * 1000)

        if json_output:
            console.print(
                json.dumps(
                    {
                        "model": settings["model"],
                        "latency_ms": latency_ms,
                        "response": response,
                    },
                    indent=2,
                )
            )
            return

        console.print(f"[bold]Model:[/bold] {settings['model']}")
        console.print(f"[bold]Latency:[/bold] {latency_ms} ms")
        console.print("\n[bold green]Response:[/bold green]")
        console.print(response)

    asyncio.run(_probe())
