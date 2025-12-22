#!/usr/bin/env python3
"""Training System Health Check.

Monitors health of:
- Running training campaigns
- Generator status
- Embedding service
- Knowledge base integrity
- System resources

Usage:
    python -m agents.training.health_check
    python -m agents.training.health_check --json
    python -m agents.training.health_check --watch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CampaignStatus:
    """Status of a running campaign."""

    pid: Optional[int]
    log_file: Optional[Path]
    running: bool
    samples_generated: int
    target_samples: int
    progress_percent: float
    current_domain: str
    samples_per_min: float
    eta_hours: float
    quality_pass_rate: float
    last_update: datetime


@dataclass
class SystemHealth:
    """Overall system health."""

    campaign: Optional[CampaignStatus]
    embedding_service_running: bool
    knowledge_bases_loaded: int
    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    api_quota_remaining: Optional[int]
    last_checkpoint: Optional[datetime]
    remote_nodes: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def find_campaign_process() -> Optional[dict]:
    """Find running campaign process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('generate_campaign' in str(arg) for arg in cmdline):
                return {
                    'pid': proc.info['pid'],
                    'cmdline': cmdline,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def parse_campaign_log(log_path: Path) -> dict:
    """Parse campaign log for progress metrics."""
    if not log_path.exists():
        return {}

    metrics = {
        'samples_generated': 0,
        'target_samples': 0,
        'current_domain': 'unknown',
        'quality_pass_rate': 0.0,
        'last_update': None,
    }

    try:
        # Read the whole file for target_samples if not found in last 500 lines
        # or just read the whole file as campaign logs are usually not huge
        with open(log_path) as f:
            lines = f.readlines()
            
        # First pass for target_samples (usually at the start)
        for line in lines:
            if 'Target:' in line and 'samples' in line:
                try:
                    parts = line.split('Target:')[1].split('samples')[0].strip()
                    metrics['target_samples'] = int(parts)
                    break
                except (ValueError, IndexError):
                    pass

        # Second pass for progress (usually at the end)
        recent_lines = lines[-500:] if len(lines) > 500 else lines
        for line in recent_lines:
            # Extract progress from "Progress: 123/6900"
            if 'Progress:' in line:
                try:
                    parts = line.split('Progress:')[1].strip().split('/')
                    if len(parts) == 2:
                        metrics['samples_generated'] = int(parts[0])
                        # If target wasn't found before, maybe it's here
                        if metrics['target_samples'] == 0:
                            metrics['target_samples'] = int(parts[1])
                except (ValueError, IndexError):
                    pass

            # Extract domain from "Generating from domain: asm"
            if 'Generating from domain:' in line:
                metrics['current_domain'] = line.split('Generating from domain:')[1].strip()

            # Extract quality pass rate from validation
            if 'Quality pass rate:' in line or 'pass rate:' in line.lower():
                # Try to find percentage
                try:
                    parts = line.split('%')[0].split()
                    if parts:
                        metrics['quality_pass_rate'] = float(parts[-1]) / 100.0
                except (ValueError, IndexError):
                    pass

        # Get last modification time
        metrics['last_update'] = datetime.fromtimestamp(log_path.stat().st_mtime)

    except Exception as e:
        logger.warning(f"Error parsing log: {e}")

    return metrics


def find_latest_campaign_log() -> Optional[Path]:
    """Find the most recent campaign log file."""
    log_dir = Path.home() / '.context' / 'logs'
    if not log_dir.exists():
        return None

    campaign_logs = list(log_dir.glob('campaign_*.log'))
    if not campaign_logs:
        return None

    # Return most recently modified
    return max(campaign_logs, key=lambda p: p.stat().st_mtime)


def get_campaign_status() -> Optional[CampaignStatus]:
    """Get status of running training campaign."""
    proc_info = find_campaign_process()
    log_path = find_latest_campaign_log()

    if not proc_info and not log_path:
        return None

    # Parse log for metrics
    log_metrics = parse_campaign_log(log_path) if log_path else {}

    running = proc_info is not None
    samples_generated = log_metrics.get('samples_generated', 0)
    target_samples = log_metrics.get('target_samples', 34500)

    progress_percent = (samples_generated / target_samples * 100) if target_samples > 0 else 0.0

    # Calculate rate from log timestamps
    samples_per_min = 0.0
    eta_hours = 0.0

    if log_path and samples_generated > 0:
        # Estimate rate from progress and time elapsed
        last_update = log_metrics.get('last_update')
        if last_update:
            # Find when campaign started (look for "Starting curation" in log)
            try:
                with open(log_path) as f:
                    for line in f:
                        if 'Starting curation' in line:
                            # Extract timestamp from log line
                            timestamp_str = line.split('[')[0].strip()
                            start_time = datetime.fromisoformat(timestamp_str.replace(',', '.'))
                            elapsed_mins = (last_update - start_time).total_seconds() / 60
                            if elapsed_mins > 0:
                                samples_per_min = samples_generated / elapsed_mins
                                remaining_samples = target_samples - samples_generated
                                eta_hours = (remaining_samples / samples_per_min / 60) if samples_per_min > 0 else 0.0
                            break
            except Exception:
                pass

    return CampaignStatus(
        pid=proc_info['pid'] if proc_info else None,
        log_file=log_path,
        running=running,
        samples_generated=samples_generated,
        target_samples=target_samples,
        progress_percent=progress_percent,
        current_domain=log_metrics.get('current_domain', 'unknown'),
        samples_per_min=samples_per_min,
        eta_hours=eta_hours,
        quality_pass_rate=log_metrics.get('quality_pass_rate', 0.0),
        last_update=log_metrics.get('last_update', datetime.now()),
    )


def check_embedding_service() -> bool:
    """Check if embedding service is running."""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('embedding' in str(arg).lower() for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def count_knowledge_bases() -> int:
    """Count loaded knowledge bases."""
    kb_dir = Path.home() / '.context' / 'knowledge'
    if not kb_dir.exists():
        return 0

    # Count directories with embeddings
    count = 0
    for item in kb_dir.iterdir():
        if item.is_dir() and (item / 'embeddings').exists():
            count += 1
    return count


def find_latest_checkpoint() -> Optional[datetime]:
    """Find the most recent training checkpoint."""
    checkpoint_dir = Path.home() / '.context' / 'training' / 'checkpoints'
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob('checkpoint_*.json'))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime)


def list_historical_campaigns() -> list[dict]:
    """List all historical training campaigns."""
    datasets_dir = Path.home() / ".context" / "training" / "datasets"
    if not datasets_dir.exists():
        return []

    campaigns = []
    for d in datasets_dir.iterdir():
        if d.is_dir():
            metadata_path = d / "metadata.json"
            stats_path = d / "stats.json"

            campaign = {
                "id": d.name,
                "path": str(d),
                "metadata": {},
                "stats": {},
            }

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        campaign["metadata"] = json.load(f)
                except Exception:
                    pass

            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        campaign["stats"] = json.load(f)
                except Exception:
                    pass

            campaigns.append(campaign)

    # Sort by creation time if available, otherwise by name
    return sorted(
        campaigns,
        key=lambda x: x["metadata"].get("created", x["id"]),
        reverse=True,
    )


async def get_remote_node_status() -> list[dict]:
    """Check status of remote inference nodes."""
    from core.nodes import node_manager

    results = []
    try:
        await node_manager.load_config()
        # Find medical-mechanica specifically or any remote gpu node
        mm_node = node_manager.get_node("medical-mechanica")
        if mm_node:
            is_healthy = await node_manager.health_check(mm_node)
            results.append(
                {
                    "name": mm_node.name,
                    "host": mm_node.host,
                    "port": mm_node.port,
                    "gpu": mm_node.metadata.get("gpu") or ("Yes" if mm_node.has_gpu else "Unknown"),
                    "memory_gb": mm_node.metadata.get("memory_gb") or (mm_node.gpu_memory_mb / 1024 if mm_node.gpu_memory_mb else "Unknown"),
                    "online": is_healthy,
                    "models": mm_node.models or [],
                }
            )
    except Exception as e:
        logger.warning(f"Error checking remote nodes: {e}")
    finally:
        await node_manager.close()

    return results


def get_system_health() -> SystemHealth:
    """Get overall system health status."""
    campaign = get_campaign_status()
    embedding_service = check_embedding_service()
    kb_count = count_knowledge_bases()

    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home()))

    issues = []

    # Check for issues
    if campaign and campaign.running:
        if campaign.samples_per_min < 5:
            issues.append(f"⚠️  Slow generation rate: {campaign.samples_per_min:.1f} samples/min (expected >10)")

        if campaign.quality_pass_rate < 0.5 and campaign.quality_pass_rate > 0:
            issues.append(f"⚠️  Low quality pass rate: {campaign.quality_pass_rate:.1%} (expected >50%)")

        # Check for stalled campaign
        if campaign.last_update and (datetime.now() - campaign.last_update) > timedelta(minutes=10):
            issues.append(f"🚨 Campaign may be stalled (last update: {campaign.last_update.strftime('%H:%M:%S')})")

    if cpu_percent > 90:
        issues.append(f"⚠️  High CPU usage: {cpu_percent:.1f}%")

    if memory.percent > 90:
        issues.append(f"⚠️  High memory usage: {memory.percent:.1f}%")

    if disk.free / (1024**3) < 10:  # Less than 10GB free
        issues.append(f"🚨 Low disk space: {disk.free / (1024**3):.1f}GB free")

    if not embedding_service:
        issues.append("ℹ️  Embedding service not running")

    return SystemHealth(
        campaign=campaign,
        embedding_service_running=embedding_service,
        knowledge_bases_loaded=kb_count,
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        disk_free_gb=disk.free / (1024**3),
        api_quota_remaining=None,
        last_checkpoint=find_latest_checkpoint(),
        remote_nodes=[],  # Will be populated in async caller
        issues=issues,
    )


async def get_system_health_async() -> SystemHealth:
    """Async version of get_system_health to handle remote node checks."""
    health = get_system_health()
    health.remote_nodes = await get_remote_node_status()
    return health


def print_health_report(health: SystemHealth):
    """Print human-readable health report."""
    print("=" * 80)
    print("TRAINING SYSTEM HEALTH CHECK")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Campaign Status
    print("\n📊 CAMPAIGN STATUS")
    if health.campaign:
        c = health.campaign
        status_icon = "🟢" if c.running else "🔴"
        print(f"  Status: {status_icon} {'Running' if c.running else 'Stopped'}")
        if c.pid:
            print(f"  PID: {c.pid}")
        if c.log_file:
            print(f"  Log: {c.log_file}")
        print(f"  Progress: {c.samples_generated:,} / {c.target_samples:,} ({c.progress_percent:.1f}%)")
        print(f"  Current Domain: {c.current_domain}")
        print(f"  Generation Rate: {c.samples_per_min:.1f} samples/min")
        print(f"  Quality Pass Rate: {c.quality_pass_rate:.1%}")
        if c.eta_hours > 0:
            print(f"  ETA: {c.eta_hours:.1f} hours")
        print(f"  Last Update: {c.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("  Status: 🔵 No active campaign")

    # System Resources
    print("\n💻 SYSTEM RESOURCES")
    cpu_icon = "🟢" if health.cpu_percent < 70 else "🟡" if health.cpu_percent < 90 else "🔴"
    print(f"  CPU: {cpu_icon} {health.cpu_percent:.1f}%")

    mem_icon = "🟢" if health.memory_percent < 70 else "🟡" if health.memory_percent < 90 else "🔴"
    print(f"  Memory: {mem_icon} {health.memory_percent:.1f}%")

    disk_icon = "🟢" if health.disk_free_gb > 50 else "🟡" if health.disk_free_gb > 10 else "🔴"
    print(f"  Disk Free: {disk_icon} {health.disk_free_gb:.1f} GB")

    # Services
    print("\n🔧 SERVICES")
    emb_icon = "🟢" if health.embedding_service_running else "🔴"
    print(f"  Embedding Service: {emb_icon} {'Running' if health.embedding_service_running else 'Stopped'}")
    print(f"  Knowledge Bases: {health.knowledge_bases_loaded} loaded")

    # Checkpoints
    if health.last_checkpoint:
        print(f"\n💾 CHECKPOINTS")
        print(f"  Last Checkpoint: {health.last_checkpoint.strftime('%Y-%m-%d %H:%M:%S')}")

    # Issues
    if health.issues:
        print(f"\n⚠️  ISSUES ({len(health.issues)})")
        for issue in health.issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ No issues detected")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Training system health check")
    parser.add_argument('--json', action='store_true', help="Output JSON format")
    parser.add_argument('--watch', action='store_true', help="Continuous monitoring (update every 60s)")
    parser.add_argument('--interval', type=int, default=60, help="Watch interval in seconds (default: 60)")

    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                if not args.json:
                    # Clear screen
                    os.system('clear' if os.name != 'nt' else 'cls')

                health = get_system_health()

                if args.json:
                    # Convert to dict, handling datetime serialization
                    health_dict = asdict(health)
                    if health_dict['campaign']:
                        health_dict['campaign']['last_update'] = health_dict['campaign']['last_update'].isoformat()
                        if health_dict['campaign']['log_file']:
                            health_dict['campaign']['log_file'] = str(health_dict['campaign']['log_file'])
                    if health_dict['last_checkpoint']:
                        health_dict['last_checkpoint'] = health_dict['last_checkpoint'].isoformat()
                    print(json.dumps(health_dict, indent=2))
                else:
                    print_health_report(health)

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        health = get_system_health()

        if args.json:
            # Convert to dict, handling datetime serialization
            health_dict = asdict(health)
            if health_dict['campaign']:
                health_dict['campaign']['last_update'] = health_dict['campaign']['last_update'].isoformat()
                if health_dict['campaign']['log_file']:
                    health_dict['campaign']['log_file'] = str(health_dict['campaign']['log_file'])
            if health_dict['last_checkpoint']:
                health_dict['last_checkpoint'] = health_dict['last_checkpoint'].isoformat()
            print(json.dumps(health_dict, indent=2))
        else:
            print_health_report(health)


if __name__ == '__main__':
    main()
