"""Training campaign monitoring commands."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pathlib import Path
import time
import sys
import json
from datetime import datetime
from typing import Optional

training_app = typer.Typer(name="training", help="Training campaign monitoring and control")
console = Console()


@training_app.command("status")
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuous monitoring"),
    interval: int = typer.Option(30, "--interval", "-i", help="Update interval in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
):
    """Show training campaign status."""
    from agents.training.health_check import get_system_health_async

    async def _status():
        health = await get_system_health_async()

        if json_output:
            from dataclasses import asdict
            health_dict = asdict(health)

            # Handle datetime serialization
            if health_dict['campaign']:
                health_dict['campaign']['last_update'] = health_dict['campaign']['last_update'].isoformat()
                if health_dict['campaign']['log_file']:
                    health_dict['campaign']['log_file'] = str(health_dict['campaign']['log_file'])
            if health_dict['last_checkpoint']:
                health_dict['last_checkpoint'] = health_dict['last_checkpoint'].isoformat()

            console.print_json(data=health_dict)
            return

        if watch:
            try:
                with Live(console=console, refresh_per_second=1) as live:
                    while True:
                        current_health = await get_system_health_async()
                        live.update(render_status(current_health))
                        time.sleep(interval)
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
        else:
            console.print(render_status(health))

    asyncio.run(_status())


@training_app.command("history")
def history(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to show"),
):
    """List historical training campaigns."""
    from agents.training.health_check import list_historical_campaigns

    campaigns = list_historical_campaigns()

    if not campaigns:
        console.print("[yellow]No historical campaigns found.[/yellow]")
        return

    table = Table(title="Training History")
    table.add_column("Run ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Domains")
    table.add_column("Samples", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Duration")

    for c in campaigns[:limit]:
        meta = c.get("metadata", {})
        stats = c.get("stats", {})

        created = meta.get("created", "unknown")
        domains = ", ".join(meta.get("domains", []))
        samples = f"{stats.get('final_count', 0)} / {meta.get('train_count', 0)}"
        quality = f"{stats.get('quality_scores', {}).get('average', 0.0):.1%}"

        duration_sec = stats.get("duration_seconds", 0)
        if duration_sec > 3600:
            duration = f"{duration_sec / 3600:.1f}h"
        elif duration_sec > 60:
            duration = f"{duration_sec / 60:.1f}m"
        else:
            duration = f"{duration_sec:.1f}s"

        table.add_row(
            c["id"],
            created,
            domains,
            samples,
            quality,
            duration
        )

    console.print(table)


@training_app.command("show")
def show(
    run_id: str = typer.Argument(..., help="Run ID (directory name)"),
):
    """Show details of a specific training run."""
    from agents.training.health_check import list_historical_campaigns

    campaigns = list_historical_campaigns()
    campaign = next((c for c in campaigns if c["id"] == run_id), None)

    if not campaign:
        console.print(f"[red]Campaign {run_id} not found.[/red]")
        raise typer.Exit(1)

    meta = campaign.get("metadata", {})
    stats = campaign.get("stats", {})

    console.print(Panel(f"[bold cyan]Run:[/bold cyan] {run_id}", expand=False))

    details = Table(show_header=False, box=None)
    details.add_column("Key", style="cyan")
    details.add_column("Value")

    details.add_row("Created", meta.get("created", "unknown"))
    details.add_row("Template", meta.get("template", "unknown"))
    details.add_row("Domains", ", ".join(meta.get("domains", [])))
    details.add_row("Target Samples", str(meta.get("train_count", 0)))
    details.add_row("Final Samples", str(stats.get("final_count", 0)))
    details.add_row("Quality Average", f"{stats.get('quality_scores', {}).get('average', 0.0):.1%}")

    duration_sec = stats.get("duration_seconds", 0)
    details.add_row("Duration", f"{duration_sec:.1f} seconds")

    console.print(details)

    if "domain_counts" in stats:
        domain_table = Table(title="Domain Distribution")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Count", justify="right")

        for domain, count in stats["domain_counts"].items():
            domain_table.add_row(domain, str(count))

        console.print(domain_table)

    # Find matching log
    from agents.training.health_check import find_latest_campaign_log
    log_dir = Path.home() / ".context" / "logs"
    # Try to find log with timestamp in it
    timestamp = meta.get("created", "")
    if timestamp:
        # Match campaign_*.log files
        matches = list(log_dir.glob(f"campaign_*{timestamp}*.log"))
        if matches:
            console.print(f"\n[blue]Associated Log:[/blue] {matches[0]}")


@training_app.command("logs")
def logs(
    run_id: Optional[str] = typer.Argument(None, help="Optional Run ID to find logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """Show training campaign logs."""
    from agents.training.health_check import find_latest_campaign_log, list_historical_campaigns

    log_path = None

    if run_id:
        campaigns = list_historical_campaigns()
        campaign = next((c for c in campaigns if c["id"] == run_id), None)
        if campaign:
            timestamp = campaign.get("metadata", {}).get("created", "")
            if timestamp:
                log_dir = Path.home() / ".context" / "logs"
                matches = list(log_dir.glob(f"campaign_*{timestamp}*.log"))
                if matches:
                    log_path = matches[0]

    if not log_path:
        log_path = find_latest_campaign_log()

    if not log_path or not log_path.exists():
        console.print("[red]No campaign log found[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Log:[/blue] {log_path}")
    console.print()

    if follow:
        # Tail -f style
        import subprocess
        try:
            subprocess.run(['tail', '-f', str(log_path)])
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped following logs.[/yellow]")
    else:
        # Show last N lines
        with open(log_path) as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                console.print(line, end='')


@training_app.command("stop")
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force stop without confirmation"),
):
    """Stop running training campaign."""
    from agents.training.health_check import find_campaign_process
    import signal

    proc_info = find_campaign_process()
    if not proc_info:
        console.print("[yellow]No running campaign found[/yellow]")
        return

    pid = proc_info['pid']

    if not force:
        confirm = typer.confirm(f"Stop campaign (PID {pid})?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        import os
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to campaign (PID {pid})[/green]")
        console.print("[blue]Checkpoints will be saved before shutdown[/blue]")
    except ProcessLookupError:
        console.print(f"[red]Process {pid} not found[/red]")
    except PermissionError:
        console.print(f"[red]Permission denied to stop process {pid}[/red]")


def render_status(health):
    """Render training status panel."""
    from datetime import datetime
    from rich.columns import Columns
    from rich import box
    from rich.panel import Panel
    from rich.console import Group

    # Title
    title = Panel.fit(
        "[bold cyan]TRAINING CAMPAIGN STATUS[/bold cyan]",
        subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
    )

    # Campaign section
    campaign_table = Table(title="Campaign", show_header=False, box=None)
    campaign_table.add_column("Key", style="cyan")
    campaign_table.add_column("Value")

    if health.campaign:
        c = health.campaign
        status_icon = "🟢" if c.running else "🔴"
        campaign_table.add_row("Status", f"{status_icon} {'Running' if c.running else 'Stopped'}")

        if c.pid:
            campaign_table.add_row("PID", str(c.pid))

        campaign_table.add_row("Progress", f"{c.samples_generated:,} / {c.target_samples:,} ({c.progress_percent:.1f}%)")
        campaign_table.add_row("Domain", c.current_domain)
        campaign_table.add_row("Rate", f"{c.samples_per_min:.1f} samples/min")
        campaign_table.add_row("Quality Pass", f"{c.quality_pass_rate:.1%}")

        if c.eta_hours > 0:
            campaign_table.add_row("ETA", f"{c.eta_hours:.1f} hours")

        campaign_table.add_row("Last Update", c.last_update.strftime("%H:%M:%S"))
    else:
        campaign_table.add_row("Status", "🔵 No active campaign")

    # System section
    system_table = Table(title="System Resources", show_header=False, box=None)
    system_table.add_column("Key", style="cyan")
    system_table.add_column("Value")

    cpu_icon = "🟢" if health.cpu_percent < 70 else "🟡" if health.cpu_percent < 90 else "🔴"
    system_table.add_row("CPU", f"{cpu_icon} {health.cpu_percent:.1f}%")

    mem_icon = "🟢" if health.memory_percent < 70 else "🟡" if health.memory_percent < 90 else "🔴"
    system_table.add_row("Memory", f"{mem_icon} {health.memory_percent:.1f}%")

    disk_icon = "🟢" if health.disk_free_gb > 50 else "🟡" if health.disk_free_gb > 10 else "🔴"
    system_table.add_row("Disk Free", f"{disk_icon} {health.disk_free_gb:.1f} GB")

    # Services section
    services_table = Table(title="Services", show_header=False, box=None)
    services_table.add_column("Key", style="cyan")
    services_table.add_column("Value")

    emb_icon = "🟢" if health.embedding_service_running else "🔴"
    services_table.add_row("Embedding Service", f"{emb_icon} {'Running' if health.embedding_service_running else 'Stopped'}")
    services_table.add_row("Knowledge Bases", str(health.knowledge_bases_loaded))

    # Remote Nodes section
    remote_table = Table(title="Remote Inference", show_header=False, box=None)
    remote_table.add_column("Key", style="cyan")
    remote_table.add_column("Value")

    if health.remote_nodes:
        for node in health.remote_nodes:
            status_icon = "🟢" if node["online"] else "🔴"
            remote_table.add_row(node["name"], f"{status_icon} {'Online' if node['online'] else 'Offline'}")
            if node["online"]:
                remote_table.add_row("  GPU", node["gpu"] or "unknown")
                remote_table.add_row("  Memory", f"{node['memory_gb']} GB")
                # Show first 3 models if many
                models = node["models"]
                model_str = ", ".join(models[:3])
                if len(models) > 3:
                    model_str += f" (+{len(models)-3} more)"
                remote_table.add_row("  Models", model_str)
    else:
        remote_table.add_row("Nodes", "None configured")

    # Windows Training section
    windows_table = Table(title="Windows Training (GPU)", show_header=False, box=None)
    windows_table.add_column("Key", style="cyan")
    windows_table.add_column("Value")

    if health.windows_training and health.windows_training.accessible:
        wt = health.windows_training
        if wt.model_name:
            status_icon = "🟢" if wt.is_running else "🔴"
            windows_table.add_row("Status", f"{status_icon} {'Running' if wt.is_running else 'Stopped'}")
            windows_table.add_row("Model", wt.model_name)
            if wt.checkpoint and wt.max_steps:
                windows_table.add_row("Progress", f"{wt.checkpoint}/{wt.max_steps} steps ({wt.progress_percent:.1f}%)")
            if wt.current_loss:
                windows_table.add_row("Current Loss", f"{wt.current_loss:.4f}")
            if wt.last_updated:
                windows_table.add_row("Last Updated", wt.last_updated.strftime("%H:%M:%S"))
        else:
            windows_table.add_row("Status", "🔵 No active training")
    else:
        windows_table.add_row("Status", "⚪ Mount not accessible")

    # Issues section
    issues_text = ""
    if health.issues:
        issues_text = f"[bold red]Issues ({len(health.issues)}):[/bold red]\n"
        for issue in health.issues:
            issues_text += f"  {issue}\n"
    else:
        issues_text = "[bold green]✅ No issues detected[/bold green]"

    # Combine all
    from rich.text import Text
    layout = Group(
        title,
        Text(""),
        Columns([campaign_table, system_table]),
        Columns([services_table, remote_table]),
        windows_table,
        Text(""),
        Text.from_markup(issues_text)
    )

    return layout


@training_app.command("qa")
def qa(
    batch_id: Optional[str] = typer.Option(None, "--batch", "-b", help="Batch ID (default: today)"),
    show_code: bool = typer.Option(True, "--code/--no-code", help="Show code context"),
):
    """Show expert questions for review and answering."""
    from agents.training.background import QuestionCurator

    curator = QuestionCurator()

    # Get batch (today's batch or specific)
    if batch_id:
        batch = curator.get_batch(batch_id)
    else:
        batch = curator.get_today_batch()

    if not batch:
        console.print("[yellow]No questions available. Run pattern analyzer first.[/yellow]")
        console.print("\n[blue]To generate questions:[/blue]")
        console.print("  hafs training qa-scan <path_to_code>")
        return

    # Show batch info
    console.print(Panel(
        f"[bold cyan]Question Batch:[/bold cyan] {batch.batch_id}\n"
        f"[cyan]Questions:[/cyan] {batch.total_count}\n"
        f"[cyan]Answered:[/cyan] {batch.answered_count}\n"
        f"[cyan]Skipped:[/cyan] {batch.skipped_count}\n"
        f"[cyan]Pending:[/cyan] {batch.pending_count}",
        title="Expert Q&A",
        expand=False,
    ))
    console.print()

    # Show each question
    for i, question in enumerate(batch.questions, 1):
        console.print(f"[bold]Question {i}/{batch.total_count}[/bold]")
        console.print(f"[cyan]ID:[/cyan] {question.question_id}")
        console.print(f"[cyan]Type:[/cyan] {question.question_type} | "
                     f"[cyan]Difficulty:[/cyan] {question.difficulty} | "
                     f"[cyan]Priority:[/cyan] {question.priority_score:.2f}")
        console.print()

        # Show question
        console.print(Panel(question.question_text, title="Question", border_style="green"))

        # Show code context if requested
        if show_code:
            pattern = question.pattern
            code_panel = f"[cyan]File:[/cyan] {pattern.file_path}\n"
            code_panel += f"[cyan]Line:[/cyan] {pattern.line_number}\n\n"
            code_panel += f"[yellow]Code:[/yellow]\n{pattern.code_snippet}\n\n"
            if pattern.context:
                code_panel += f"[yellow]Context:[/yellow]\n{pattern.context[:500]}"

            console.print(Panel(code_panel, title="Code Context", border_style="blue"))

        console.print()

    # Show usage instructions
    console.print(Panel(
        f"[bold cyan]Commands:[/bold cyan]\n\n"
        f"AI-assisted answer (recommended for mobile):\n"
        f"  hafs training qa-assist <question_id>\n\n"
        f"Manual answer:\n"
        f"  hafs training qa-answer <question_id>\n\n"
        f"Skip a question:\n"
        f"  hafs training qa-skip <question_id>\n\n"
        f"View statistics:\n"
        f"  hafs training qa-stats",
        title="Next Steps",
        border_style="cyan",
    ))


@training_app.command("qa-assist")
def qa_assist(
    question_id: str = typer.Argument(..., help="Question ID"),
):
    """AI-assisted answer using your code, git history, and web search (mobile-friendly)."""
    from agents.training.background.assisted_qa import assisted_answer_workflow

    console.print("[cyan]Generating draft answer using:[/cyan]")
    console.print("  • Your code implementation")
    console.print("  • Git commit history")
    console.print("  • Web search for technical context")
    console.print()

    async def _assist():
        try:
            result = await assisted_answer_workflow(question_id)

            console.print(Panel(
                f"[bold]Draft Answer[/bold]\n\n{result['draft_answer']}\n\n"
                f"[dim]Sources: {', '.join(result['sources'])}[/dim]\n"
                f"[dim]Confidence: {result['confidence']}[/dim]",
                title="Generated Draft",
                border_style="cyan",
            ))
            console.print()

            # Ask if they want to accept, edit, or skip
            console.print("[bold]Options:[/bold]")
            console.print("  [green]a[/green] - Accept and save (will generate 3-5 training samples)")
            console.print("  [yellow]e[/yellow] - Edit before saving")
            console.print("  [red]s[/red] - Skip this question")
            console.print()

            choice = input("Choice [a/e/s]: ").lower().strip()

            if choice == 'a':
                # Accept draft
                from agents.training.background import QuestionCurator, QAConverter

                curator = QuestionCurator()
                answered = curator.answer_question(question_id, result['draft_answer'])
                console.print(f"[green]✓ Answer saved ({answered.answer_word_count} words)[/green]")

                # Convert to samples
                console.print("\n[blue]Converting to training samples...[/blue]")
                converter = QAConverter()
                await converter.setup()
                samples = await converter.convert_qa_to_samples(answered, num_variations=3)

                if samples:
                    output_dir = Path.home() / ".context" / "training" / "qa_samples"
                    output_path = output_dir / f"{question_id}.jsonl"
                    await converter.save_samples(samples, output_path)

                    console.print(f"[green]✓ Generated {len(samples)} training samples[/green]")
                    console.print(f"[cyan]Saved to:[/cyan] {output_path}")

            elif choice == 'e':
                console.print("\n[cyan]Paste/edit the answer below (press Ctrl+D when done):[/cyan]")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                edited_answer = "\n".join(lines)

                if edited_answer.strip():
                    from agents.training.background import QuestionCurator, QAConverter

                    curator = QuestionCurator()
                    answered = curator.answer_question(question_id, edited_answer.strip())
                    console.print(f"[green]✓ Answer saved ({answered.answer_word_count} words)[/green]")

                    # Convert to samples
                    console.print("\n[blue]Converting to training samples...[/blue]")
                    converter = QAConverter()
                    await converter.setup()
                    samples = await converter.convert_qa_to_samples(answered, num_variations=3)

                    if samples:
                        output_dir = Path.home() / ".context" / "training" / "qa_samples"
                        output_path = output_dir / f"{question_id}.jsonl"
                        await converter.save_samples(samples, output_path)

                        console.print(f"[green]✓ Generated {len(samples)} training samples[/green]")

            elif choice == 's':
                from agents.training.background import QuestionCurator
                curator = QuestionCurator()
                curator.skip_question(question_id)
                console.print("[yellow]✓ Question skipped[/yellow]")

            else:
                console.print("[red]Invalid choice. Cancelled.[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

    asyncio.run(_assist())


@training_app.command("qa-answer")
def qa_answer(
    question_id: str = typer.Argument(..., help="Question ID"),
    answer: Optional[str] = typer.Option(None, "--answer", "-a", help="Answer text (or will prompt)"),
):
    """Answer an expert question manually (type your own answer)."""
    from agents.training.background import QuestionCurator, QAConverter

    curator = QuestionCurator()

    # Find question
    all_questions = curator.load_all_questions()
    question = next((q for q in all_questions if q.question_id == question_id), None)

    if not question:
        console.print(f"[red]Question not found: {question_id}[/red]")
        raise typer.Exit(1)

    # Show question
    console.print(Panel(question.question_text, title="Question", border_style="green"))
    console.print()

    # Get answer
    if not answer:
        console.print("[cyan]Enter your answer (press Ctrl+D when done):[/cyan]")
        console.print("[dim]Tip: Use 'hafs training qa-assist' for AI-assisted answering[/dim]\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        answer = "\n".join(lines)

    if not answer or not answer.strip():
        console.print("[yellow]No answer provided. Cancelled.[/yellow]")
        return

    # Save answer
    answered = curator.answer_question(question_id, answer.strip())
    console.print(f"[green]✓ Answer saved ({answered.answer_word_count} words)[/green]")

    # Convert to training samples
    console.print("\n[blue]Converting answer to training samples...[/blue]")

    async def _convert():
        converter = QAConverter()
        await converter.setup()
        samples = await converter.convert_qa_to_samples(answered, num_variations=3)

        if samples:
            # Save samples
            output_dir = Path.home() / ".context" / "training" / "qa_samples"
            output_path = output_dir / f"{question_id}.jsonl"
            await converter.save_samples(samples, output_path)

            console.print(f"[green]✓ Generated {len(samples)} training samples[/green]")
            console.print(f"[cyan]Saved to:[/cyan] {output_path}")
        else:
            console.print("[yellow]⚠ Failed to generate samples[/yellow]")

    asyncio.run(_convert())


@training_app.command("qa-skip")
def qa_skip(
    question_id: str = typer.Argument(..., help="Question ID"),
):
    """Skip an expert question."""
    from agents.training.background import QuestionCurator

    curator = QuestionCurator()
    curator.skip_question(question_id)

    console.print(f"[green]✓ Skipped question: {question_id}[/green]")


@training_app.command("qa-stats")
def qa_stats():
    """Show Q&A statistics."""
    from agents.training.background import QuestionCurator

    curator = QuestionCurator()
    stats = curator.get_statistics()

    # Create stats table
    table = Table(title="Expert Q&A Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Questions", str(stats["total_questions"]))
    table.add_row("Answered", f"[green]{stats['answered_count']}[/green]")
    table.add_row("Skipped", f"[yellow]{stats['skipped_count']}[/yellow]")
    table.add_row("Pending", f"[blue]{stats['pending_count']}[/blue]")

    if stats["answered_count"] > 0:
        table.add_row("", "")  # Separator
        table.add_row("Avg Answer Length", f"{stats['answer_word_count_avg']:.0f} words")
        table.add_row("Conversion Rate", f"{stats['conversion_rate']:.1%}")
        table.add_row("Total Samples Generated", str(stats["total_samples_generated"]))

    console.print(table)

    # Show progress bar
    if stats["total_questions"] > 0:
        completion = (stats["answered_count"] + stats["skipped_count"]) / stats["total_questions"]
        console.print()
        console.print(f"[cyan]Overall Progress:[/cyan] {completion:.1%}")


@training_app.command("qa-scan")
def qa_scan(
    path: str = typer.Argument(..., help="Path to codebase to scan"),
    patterns: Optional[str] = typer.Option(None, "--patterns", "-p", help="File patterns (comma-separated)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max questions to generate"),
):
    """Scan codebase and generate expert questions."""
    from agents.training.background import PatternAnalyzerAgent, QuestionCurator
    from pathlib import Path

    scan_path = Path(path)
    if not scan_path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    async def _scan():
        # Initialize analyzer
        console.print(f"[cyan]Scanning {scan_path} for code patterns...[/cyan]")
        agent = PatternAnalyzerAgent()
        await agent.setup()

        # Parse patterns
        file_patterns = None
        if patterns:
            file_patterns = [p.strip() for p in patterns.split(",")]

        # Scan codebase
        detected = await agent.analyze_codebase(scan_path, file_patterns)
        console.print(f"[green]✓ Found {len(detected)} code patterns[/green]")

        # Rank by pedagogical value
        ranked = sorted(
            detected,
            key=lambda p: p.pedagogical_value * p.complexity_score,
            reverse=True,
        )[:limit]

        console.print(f"[cyan]Generating questions for top {len(ranked)} patterns...[/cyan]")

        # Generate questions
        questions = []
        with console.status("[cyan]Generating...") as status:
            for i, pattern in enumerate(ranked, 1):
                status.update(f"[cyan]Generating question {i}/{len(ranked)}...[/cyan]")
                question = await agent.generate_question(pattern)
                if question:
                    questions.append(question)

        console.print(f"[green]✓ Generated {len(questions)} questions[/green]")

        # Save questions
        if questions:
            await agent.save_questions(questions)
            console.print(f"[cyan]Saved to:[/cyan] {agent.questions_db}")

            # Create today's batch
            curator = QuestionCurator()
            batch = curator.create_batch()
            if batch:
                console.print(f"\n[green]✓ Created batch {batch.batch_id} with {batch.total_count} questions[/green]")
                console.print("\n[blue]Review questions:[/blue]")
                console.print("  hafs training qa")

    asyncio.run(_scan())


if __name__ == "__main__":
    training_app()
