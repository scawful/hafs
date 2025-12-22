"""Console UI for orchestration commands."""
from rich.console import Console
from typing import Any

def render_orchestration_result(console: Console, result: Any) -> None:
    """Render orchestration result."""
    if result:
        console.print(result)
