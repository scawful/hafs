"""Model registry and deployment commands."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from hafs.training.model_registry import ModelRegistry, ModelLocation, ServingBackend
from hafs.training.model_deployment import ModelDeployment

models_app = typer.Typer(name="models", help="Model registry and deployment")
console = Console()


@models_app.command("list")
def list_models(
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Filter by role"),
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Filter by location"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Filter by serving backend"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
):
    """List registered models."""
    registry = ModelRegistry()

    models = registry.list_models(
        role=role,
        location=location,
        backend=backend,
    )

    if json_output:
        import json
        data = [model.to_dict() for model in models]
        console.print_json(data={"models": data, "total": len(data)})
        return

    if not models:
        console.print("[yellow]No models found matching filters[/yellow]")
        return

    # Create table
    table = Table(title=f"Registered Models ({len(models)})")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="green")
    table.add_column("Role", style="yellow")
    table.add_column("Base Model", style="blue")
    table.add_column("Locations", style="magenta")
    table.add_column("Backends", style="bright_green")

    for model in models:
        locations_str = ", ".join(model.locations.keys())
        backends_str = ", ".join(model.deployed_backends) if model.deployed_backends else "-"

        table.add_row(
            model.model_id,
            model.display_name,
            model.role,
            model.base_model,
            locations_str,
            backends_str,
        )

    console.print(table)


@models_app.command("info")
def model_info(model_id: str):
    """Show detailed model information."""
    registry = ModelRegistry()

    model = registry.get_model(model_id)
    if not model:
        console.print(f"[red]✗[/red] Model {model_id} not found")
        raise typer.Exit(1)

    # Display info
    console.print(Panel(f"[bold]{model.display_name}[/bold]", title="Model Info"))

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Model ID", model.model_id)
    info_table.add_row("Version", model.version)
    info_table.add_row("Role", model.role)
    info_table.add_row("Group", model.group)
    info_table.add_row("Base Model", model.base_model)
    info_table.add_row("Training Date", model.training_date)
    info_table.add_row("Duration", f"{model.training_duration_minutes} minutes")

    console.print(info_table)
    console.print()

    # Dataset info
    console.print("[bold]Dataset:[/bold]")
    dataset_table = Table(show_header=False, box=None)
    dataset_table.add_column("Key", style="cyan")
    dataset_table.add_column("Value", style="white")

    dataset_table.add_row("Name", model.dataset_name)
    dataset_table.add_row("Samples", f"{model.train_samples} train / {model.val_samples} val / {model.test_samples} test")

    if model.dataset_quality:
        for key, value in model.dataset_quality.items():
            dataset_table.add_row(f"  {key}", f"{value:.2%}" if isinstance(value, float) and value < 1 else str(value))

    console.print(dataset_table)
    console.print()

    # Metrics
    console.print("[bold]Metrics:[/bold]")
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Key", style="cyan")
    metrics_table.add_column("Value", style="white")

    if model.final_loss:
        metrics_table.add_row("Final Loss", f"{model.final_loss:.4f}")
    if model.best_loss:
        metrics_table.add_row("Best Loss", f"{model.best_loss:.4f}")
    if model.perplexity:
        metrics_table.add_row("Perplexity", f"{model.perplexity:.2f}")

    console.print(metrics_table)
    console.print()

    # Locations
    console.print("[bold]Locations:[/bold]")
    for location, path in model.locations.items():
        primary = " (primary)" if location == model.primary_location else ""
        console.print(f"  • {location}{primary}: {path}")

    console.print()

    # Backends
    if model.deployed_backends:
        console.print("[bold]Deployed To:[/bold]")
        for backend in model.deployed_backends:
            console.print(f"  • {backend}")
            if backend == "ollama" and model.ollama_model_name:
                console.print(f"    Name: {model.ollama_model_name}")
            elif backend == "halext-node" and model.halext_node_id:
                console.print(f"    Node: {model.halext_node_id}")


@models_app.command("pull")
def pull_model(
    model_id: str,
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source location (windows, cloud, etc.)"),
    dest: Optional[str] = typer.Option(None, "--dest", "-d", help="Local destination directory"),
):
    """Pull model from remote machine to local."""
    console.print(f"[cyan]Pulling model {model_id}...[/cyan]")

    deployment = ModelDeployment()

    try:
        dest_path = Path(dest) if dest else None
        pulled_path = deployment.pull_model(
            model_id=model_id,
            source_location=source,
            destination=dest_path,
        )

        console.print(f"[green]✓[/green] Model pulled to {pulled_path}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to pull model: {e}")
        raise typer.Exit(1)


@models_app.command("deploy")
def deploy_model(
    model_id: str,
    backend: str = typer.Argument(..., help="Backend: ollama, llama.cpp, halext"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Model name in backend"),
    quantization: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization for GGUF (Q4_K_M, Q5_K_M, Q8_0, etc.)"),
):
    """Deploy model to serving backend."""
    console.print(f"[cyan]Deploying {model_id} to {backend}...[/cyan]")

    deployment = ModelDeployment()

    try:
        if backend == "ollama":
            deployed_name = deployment.deploy_to_ollama(
                model_id=model_id,
                ollama_model_name=name,
                quantization=quantization,
            )
            console.print(f"[green]✓[/green] Deployed to Ollama as '{deployed_name}'")

        elif backend == "halext":
            if not name:
                console.print("[red]✗[/red] --name required for halext deployment")
                raise typer.Exit(1)

            # For halext, name is node URL
            node_url = name
            node_name = model_id

            deployed_name = deployment.deploy_to_halext(
                model_id=model_id,
                node_name=node_name,
                node_url=node_url,
            )
            console.print(f"[green]✓[/green] Deployed to halext node '{deployed_name}'")

        else:
            console.print(f"[red]✗[/red] Unknown backend: {backend}")
            console.print("Supported backends: ollama, halext")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Deployment failed: {e}")
        raise typer.Exit(1)


@models_app.command("test")
def test_model(
    model_id: str,
    backend: str = typer.Argument(..., help="Backend to test: ollama, llama.cpp, halext"),
):
    """Test deployed model."""
    console.print(f"[cyan]Testing {model_id} on {backend}...[/cyan]")

    deployment = ModelDeployment()

    try:
        result = deployment.test_model(model_id=model_id, backend=backend)

        if result["status"] == "success":
            console.print(f"[green]✓[/green] Test passed")
            console.print(f"\nPrompt: {result['prompt']}")
            console.print(f"\nResponse:\n{result['response']}")

        else:
            console.print(f"[red]✗[/red] Test failed: {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Test failed: {e}")
        raise typer.Exit(1)


@models_app.command("convert")
def convert_model(
    model_id: str,
    format: str = typer.Argument(..., help="Target format: gguf"),
    quantization: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization (for GGUF)"),
):
    """Convert model to different format."""
    console.print(f"[cyan]Converting {model_id} to {format}...[/cyan]")

    deployment = ModelDeployment()

    try:
        if format.lower() == "gguf":
            output_path = deployment.convert_to_gguf(
                model_id=model_id,
                quantization=quantization,
            )
            console.print(f"[green]✓[/green] Converted to GGUF: {output_path}")

        else:
            console.print(f"[red]✗[/red] Unknown format: {format}")
            console.print("Supported formats: gguf")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗[/red] Conversion failed: {e}")
        raise typer.Exit(1)


@models_app.command("register")
def register_model(
    model_path: str = typer.Argument(..., help="Path to model directory"),
    model_id: Optional[str] = typer.Option(None, "--id", help="Model ID (auto-detect from path if not provided)"),
    display_name: Optional[str] = typer.Option(None, "--name", "-n", help="Display name"),
    base_model: str = typer.Option(..., "--base", "-b", help="Base model (e.g., Qwen/Qwen2.5-Coder-1.5B)"),
    role: str = typer.Option(..., "--role", "-r", help="Model role (general, asm, debug, yaze)"),
    location: str = typer.Option("mac", "--location", "-l", help="Location (mac, windows, cloud)"),
):
    """Register a trained model in the registry."""
    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        console.print(f"[red]✗[/red] Model path not found: {model_path}")
        raise typer.Exit(1)

    # Auto-detect model_id from path if not provided
    if not model_id:
        model_id = model_path_obj.name

    if not display_name:
        display_name = model_id

    registry = ModelRegistry()

    try:
        metadata = registry.register_model(
            model_id=model_id,
            display_name=display_name,
            base_model=base_model,
            role=role,
            model_path=str(model_path_obj),
            location=location,
            group="rom-tooling",  # Default
            training_date="",
            training_duration_minutes=0,
            dataset_name="unknown",
            dataset_path="",
            train_samples=0,
            val_samples=0,
            test_samples=0,
            dataset_quality={},
            lora_config={},
            hyperparameters={},
            hardware="unknown",
            device="unknown",
        )

        console.print(f"[green]✓[/green] Registered model: {model_id}")
        console.print(f"  Display Name: {display_name}")
        console.print(f"  Location: {location}:{model_path}")

    except Exception as e:
        console.print(f"[red]✗[/red] Registration failed: {e}")
        raise typer.Exit(1)
