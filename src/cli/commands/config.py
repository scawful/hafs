"""Configuration management CLI commands."""

import json
import tomllib
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from core.config.loader import CognitiveProtocolConfig, ConfigLoader, get_config

config_app = typer.Typer(
    name="config",
    help="Manage cognitive protocol configuration",
)
console = Console()


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@config_app.command("show")
def show(
    personality: Optional[str] = typer.Option(
        None, "--personality", "-p", help="Personality to show (if not provided, shows default)"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", "-s", help="Show specific section (e.g., 'metacognition', 'emotions')"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, toml"
    ),
) -> None:
    """Display current cognitive protocol configuration."""
    try:
        cfg = get_config(personality=personality, reload=True)

        # Get the section to display
        if section:
            section_lower = section.lower()
            if hasattr(cfg, section_lower):
                data = getattr(cfg, section_lower).model_dump()
                title = f"Configuration: {section}"
            else:
                console.print(f"[red]Error:[/red] Unknown section '{section}'")
                console.print(f"Available sections: {', '.join([f.name for f in cfg.model_fields.keys()])}")
                raise typer.Exit(1)
        else:
            data = cfg.model_dump()
            title = "Cognitive Protocol Configuration"

        if personality:
            title += f" (Personality: {personality})"

        # Format output
        if format.lower() == "json":
            json_str = json.dumps(data, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=title, border_style="blue"))
        elif format.lower() == "toml":
            # Convert dict to TOML-like string representation
            import io
            output = io.StringIO()
            _dict_to_toml(data, output)
            toml_str = output.getvalue()
            syntax = Syntax(toml_str, "toml", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=title, border_style="blue"))
        else:
            console.print(f"[red]Error:[/red] Unknown format '{format}'. Use 'json' or 'toml'.")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)


@config_app.command("validate")
def validate(
    file: Path = typer.Argument(..., help="TOML file to validate"),
) -> None:
    """Validate a cognitive protocol TOML configuration file."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    try:
        # Load and parse TOML
        with open(file, "rb") as f:
            toml_data = tomllib.load(f)

        console.print(f"[green]✓[/green] Valid TOML syntax")

        # Try to validate against Pydantic schema
        try:
            cfg = CognitiveProtocolConfig.model_validate(toml_data)
            console.print(f"[green]✓[/green] Valid cognitive protocol configuration")
            console.print(f"\n[bold]Configuration Summary:[/bold]")
            console.print(f"  • Spinning threshold: {cfg.metacognition.spinning_threshold}")
            console.print(f"  • Max items in focus: {cfg.metacognition.max_items_in_focus}")
            console.print(f"  • Uncertainty threshold: {cfg.metacognition.help_seeking.uncertainty_threshold}")
            console.print(f"  • Max golden facts: {cfg.epistemic.max_golden_facts}")
            console.print(f"  • Conflict patterns: {len(cfg.goals.conflict_patterns)}")
        except ValidationError as e:
            console.print(f"[red]✗[/red] Validation errors found:")
            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])
                console.print(f"  • [yellow]{field}[/yellow]: {err['msg']}")
            raise typer.Exit(1)

    except tomllib.TOMLDecodeError as e:
        console.print(f"[red]✗[/red] Invalid TOML syntax: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_app.command("list-personalities")
def list_personalities(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """List available personality profiles."""
    try:
        # Load personality config
        config_dir = Path.cwd() / "config"
        personality_file = config_dir / "agent_personalities.toml"

        if not personality_file.exists():
            console.print(f"[yellow]Warning:[/yellow] Personality config not found at {personality_file}")
            console.print("No custom personalities configured.")
            raise typer.Exit(0)

        with open(personality_file, "rb") as f:
            personalities_data = tomllib.load(f)

        if "personalities" not in personalities_data:
            console.print("No personalities found in configuration.")
            raise typer.Exit(0)

        personalities = personalities_data["personalities"]

        if verbose:
            # Detailed view
            for name, data in personalities.items():
                description = data.get("description", "No description available")
                console.print(Panel(
                    f"[bold]{name}[/bold]\n{description}",
                    border_style="cyan"
                ))

                # Show some key differences from default
                if "metacognition" in data:
                    meta = data["metacognition"]
                    console.print("\n[dim]Key Configuration:[/dim]")
                    if "spinning_threshold" in meta:
                        console.print(f"  • Spinning threshold: {meta['spinning_threshold']}")
                    if "help_seeking" in meta and "uncertainty_threshold" in meta["help_seeking"]:
                        console.print(f"  • Uncertainty threshold: {meta['help_seeking']['uncertainty_threshold']}")
                console.print()
        else:
            # Table view
            table = Table(title="Available Personality Profiles")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Description")

            for name, data in personalities.items():
                description = data.get("description", "No description")
                table.add_row(name, description)

            console.print(table)
            console.print("\n[dim]Use --verbose to see detailed configuration for each personality[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_app.command("compare")
def compare(
    personality1: str = typer.Argument(..., help="First personality to compare"),
    personality2: str = typer.Argument(..., help="Second personality to compare"),
) -> None:
    """Compare two personality profiles."""
    try:
        cfg1 = get_config(personality=personality1, reload=True)
        cfg2 = get_config(personality=personality2, reload=True)

        console.print(Panel(
            f"Comparing [cyan]{personality1}[/cyan] vs [cyan]{personality2}[/cyan]",
            style="bold"
        ))

        # Compare key settings
        table = Table(title="Key Differences")
        table.add_column("Setting", style="yellow")
        table.add_column(personality1, style="cyan")
        table.add_column(personality2, style="magenta")
        table.add_column("Diff", style="dim")

        comparisons = [
            ("Spinning Threshold",
             cfg1.metacognition.spinning_threshold,
             cfg2.metacognition.spinning_threshold),
            ("Max Items in Focus",
             cfg1.metacognition.max_items_in_focus,
             cfg2.metacognition.max_items_in_focus),
            ("Uncertainty Threshold",
             cfg1.metacognition.help_seeking.uncertainty_threshold,
             cfg2.metacognition.help_seeking.uncertainty_threshold),
            ("Cognitive Load Warning",
             cfg1.metacognition.cognitive_load_warning,
             cfg2.metacognition.cognitive_load_warning),
            ("Max Frustration (Flow)",
             cfg1.metacognition.flow_state.max_frustration,
             cfg2.metacognition.flow_state.max_frustration),
            ("Edits Without Tests",
             cfg1.analysis_triggers.edits_without_tests,
             cfg2.analysis_triggers.edits_without_tests),
        ]

        for setting, val1, val2 in comparisons:
            diff = ""
            if val1 != val2:
                diff_val = val1 - val2 if isinstance(val1, (int, float)) else "different"
                diff = f"{diff_val:+}" if isinstance(diff_val, (int, float)) else diff_val
            table.add_row(setting, str(val1), str(val2), diff)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _dict_to_toml(data: dict, output, prefix: str = "") -> None:
    """Convert dictionary to TOML-like string representation."""
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Check if it's a simple dict (all values are primitives)
            if all(not isinstance(v, dict) for v in value.values()):
                output.write(f"\n[{full_key}]\n")
                for k, v in value.items():
                    if isinstance(v, list):
                        output.write(f"{k} = {json.dumps(v)}\n")
                    elif isinstance(v, str):
                        output.write(f'{k} = "{v}"\n')
                    else:
                        output.write(f"{k} = {v}\n")
            else:
                # Nested dict - recurse
                _dict_to_toml(value, output, full_key)
        elif isinstance(value, list):
            output.write(f"{key} = {json.dumps(value)}\n")
        elif isinstance(value, str):
            output.write(f'{key} = "{value}"\n')
        else:
            output.write(f"{key} = {value}\n")
