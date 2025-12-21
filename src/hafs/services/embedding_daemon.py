"""Embedding Daemon Service.

Runs continuously to generate embeddings for all registered projects.
Designed to run as a background service via launchd or systemd.

Usage:
    # Run directly
    python -m hafs.services.embedding_daemon

    # With options
    python -m hafs.services.embedding_daemon --batch-size 50 --interval 30

    # Install as launchd service
    python -m hafs.services.embedding_daemon --install

    # Check status
    python -m hafs.services.embedding_daemon --status
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from hafs.core.config import hafs_config
from hafs.core.runtime import resolve_python_executable

# Configure logging
LOG_DIR = Path.home() / ".context" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "embedding_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class EmbeddingDaemon:
    """Daemon for continuous embedding generation."""

    def __init__(
        self,
        batch_size: int = 50,
        interval_seconds: int = 60,
        max_daily: int = 5000,
    ):
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self.max_daily = max_daily

        # State
        self._running = False
        self._daily_count = 0
        self._daily_reset = datetime.now().date()

        # Paths
        self.data_dir = Path.home() / ".context" / "embedding_service"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.data_dir / "daemon.pid"
        self.status_file = self.data_dir / "daemon_status.json"
        self.post_state_file = self.data_dir / "post_completion.json"

        self._post_state = {
            "last_generated_at": None,
            "last_trigger_at": None,
        }
        self._post_completion_enabled = False
        self._post_completion_mode = "swarm"
        self._post_completion_topic = "Refresh knowledge after embeddings complete."
        self._post_completion_cooldown_minutes = 240
        self._post_completion_context_burst = True
        self._post_completion_context_force = True

        # Components (lazy loaded)
        self._orchestrator = None
        self._kb = None

        self._load_post_completion_state()
        self._apply_post_completion_policy()

    async def start(self):
        """Start the daemon."""
        logger.info("Starting embedding daemon...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Interval: {self.interval_seconds}s")
        logger.info(f"  Max daily: {self.max_daily}")
        logger.info(
            "  Post-completion: enabled=%s mode=%s cooldown=%sm context_burst=%s",
            self._post_completion_enabled,
            self._post_completion_mode,
            self._post_completion_cooldown_minutes,
            self._post_completion_context_burst,
        )

        # Write PID file
        self.pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._running = True
        await self._run_loop()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def _ensure_orchestrator(self):
        """Initialize orchestrator."""
        if self._orchestrator is None:
            try:
                from hafs.core.orchestrator_v2 import UnifiedOrchestrator
                self._orchestrator = UnifiedOrchestrator(log_thoughts=False)
                await self._orchestrator.initialize()
                logger.info("Orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize orchestrator: {e}")
                raise

    async def _ensure_kb(self):
        """Initialize knowledge base."""
        if self._kb is None:
            try:
                from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase
                self._kb = ALTTPKnowledgeBase()
                await self._kb.setup()
                logger.info("Knowledge base initialized")
            except Exception as e:
                logger.error(f"Failed to initialize KB: {e}")
                raise

    async def _run_loop(self):
        """Main daemon loop."""
        while self._running:
            try:
                # Reset daily count at midnight
                today = datetime.now().date()
                if today != self._daily_reset:
                    self._daily_count = 0
                    self._daily_reset = today
                    logger.info("Daily count reset")

                # Check daily limit
                if self._daily_count >= self.max_daily:
                    logger.info(f"Daily limit reached ({self.max_daily}), sleeping...")
                    await self._sleep_until_midnight()
                    continue

                # Run a batch
                generated = await self._run_batch()
                self._daily_count += generated
                if generated:
                    self._record_progress()
                else:
                    await self._maybe_trigger_post_completion()

                # Update status
                self._update_status(generated)

                if generated == 0:
                    # No more to generate, sleep longer
                    logger.info("No items to generate, sleeping 5 minutes...")
                    await asyncio.sleep(300)
                else:
                    # Normal interval
                    await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                await asyncio.sleep(60)

        # Cleanup
        self._cleanup()

    def _parse_bool(self, value: Optional[str]) -> Optional[bool]:
        """Parse truthy/falsy env values."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    def _apply_post_completion_policy(self) -> None:
        """Apply post-completion policy from config/env."""
        try:
            config = hafs_config.embedding_daemon
        except Exception:
            config = None

        if config:
            self._post_completion_enabled = bool(config.post_completion_enabled)
            self._post_completion_mode = config.post_completion_mode
            self._post_completion_topic = config.post_completion_topic
            self._post_completion_cooldown_minutes = int(
                config.post_completion_cooldown_minutes
            )
            self._post_completion_context_burst = bool(
                config.post_completion_context_burst
            )
            self._post_completion_context_force = bool(
                config.post_completion_context_force
            )

        env_enabled = self._parse_bool(os.environ.get("HAFS_EMBEDDING_POST_ENABLED"))
        if env_enabled is not None:
            self._post_completion_enabled = env_enabled
        env_mode = os.environ.get("HAFS_EMBEDDING_POST_MODE")
        if env_mode:
            self._post_completion_mode = env_mode.strip().lower()
        env_topic = os.environ.get("HAFS_EMBEDDING_POST_TOPIC")
        if env_topic:
            self._post_completion_topic = env_topic.strip()
        env_cooldown = os.environ.get("HAFS_EMBEDDING_POST_COOLDOWN_MINUTES")
        if env_cooldown:
            try:
                self._post_completion_cooldown_minutes = int(env_cooldown)
            except ValueError:
                logger.warning("Invalid HAFS_EMBEDDING_POST_COOLDOWN_MINUTES=%s", env_cooldown)
        env_context_burst = self._parse_bool(
            os.environ.get("HAFS_EMBEDDING_POST_CONTEXT_BURST")
        )
        if env_context_burst is not None:
            self._post_completion_context_burst = env_context_burst
        env_context_force = self._parse_bool(
            os.environ.get("HAFS_EMBEDDING_POST_CONTEXT_FORCE")
        )
        if env_context_force is not None:
            self._post_completion_context_force = env_context_force

    def _load_post_completion_state(self) -> None:
        """Load post-completion state from disk."""
        if not self.post_state_file.exists():
            return
        try:
            data = json.loads(self.post_state_file.read_text())
            if isinstance(data, dict):
                self._post_state.update(data)
        except Exception:
            logger.warning("Failed to load post-completion state")

    def _save_post_completion_state(self) -> None:
        """Persist post-completion state to disk."""
        try:
            self.post_state_file.write_text(json.dumps(self._post_state, indent=2))
        except Exception as e:
            logger.error(f"Failed to save post-completion state: {e}")

    def _record_progress(self) -> None:
        """Record that embedding generation made progress."""
        self._post_state["last_generated_at"] = datetime.now().isoformat()
        self._save_post_completion_state()

    def _parse_iso_time(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    async def _maybe_trigger_post_completion(self) -> None:
        """Run post-completion actions if embeddings appear caught up."""
        if not self._post_completion_enabled:
            return

        last_generated = self._parse_iso_time(self._post_state.get("last_generated_at"))
        if not last_generated:
            return

        last_trigger = self._parse_iso_time(self._post_state.get("last_trigger_at"))
        if last_trigger and last_trigger >= last_generated:
            return

        if last_trigger:
            cooldown_seconds = self._post_completion_cooldown_minutes * 60
            elapsed = (datetime.now() - last_trigger).total_seconds()
            if elapsed < cooldown_seconds:
                return

        await self._trigger_post_completion()
        self._post_state["last_trigger_at"] = datetime.now().isoformat()
        self._save_post_completion_state()

    def _context_daemon_running(self) -> bool:
        pid_file = Path.home() / ".context" / "context_agent_daemon" / "daemon.pid"
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):
            return False

    def _request_context_burst(self, force: bool) -> None:
        data_dir = Path.home() / ".context" / "context_agent_daemon"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "requested_at": datetime.now().isoformat(),
            "force": force,
        }
        (data_dir / "burst_request.json").write_text(json.dumps(payload, indent=2))

    async def _trigger_context_burst(self) -> None:
        """Kick off context agent burst work."""
        if self._context_daemon_running():
            self._request_context_burst(self._post_completion_context_force)
            return

        try:
            from hafs.services.context_agent_daemon import ContextAgentDaemon

            daemon = ContextAgentDaemon()
            await daemon.run_burst(force=self._post_completion_context_force)
        except Exception as e:
            logger.error(f"Failed to run context burst: {e}")

    async def _trigger_post_completion(self) -> None:
        """Run post-completion swarm/context actions."""
        logger.info("Embeddings appear complete; running post-completion actions")

        try:
            from hafs.core.orchestration_entrypoint import run_orchestration

            await run_orchestration(
                mode=self._post_completion_mode,
                topic=self._post_completion_topic,
            )
        except Exception as e:
            logger.error(f"Post-completion orchestration failed: {e}")

        if self._post_completion_context_burst:
            await self._trigger_context_burst()

    async def notify_embeddings_complete(self, force: bool = False) -> bool:
        """Notify the daemon that embeddings just completed."""
        if not self._post_completion_enabled:
            return False

        if force:
            self._post_state["last_generated_at"] = datetime.now().isoformat()
            await self._trigger_post_completion()
            self._post_state["last_trigger_at"] = datetime.now().isoformat()
            self._save_post_completion_state()
            return True

        self._record_progress()
        await self._maybe_trigger_post_completion()
        return True

    async def _run_batch(self) -> int:
        """Run a single batch of embedding generation."""
        try:
            await self._ensure_orchestrator()
            await self._ensure_kb()

            existing = set(self._kb._embeddings.keys())
            generated = 0
            errors = 0

            # First: symbols with descriptions
            missing_symbols = [
                s for s in self._kb._symbols.values()
                if s.id not in existing and s.description
            ]

            if missing_symbols:
                batch = missing_symbols[:self.batch_size]
                logger.info(f"Processing {len(batch)} symbols...")

                for symbol in batch:
                    try:
                        text = f"{symbol.name}: {symbol.description}"
                        result = await self._orchestrator.embed(text)
                        if result:
                            self._kb._embeddings[symbol.id] = result
                            emb_file = self._kb.embeddings_dir / f"{hashlib.md5(symbol.id.encode()).hexdigest()[:12]}.json"
                            emb_file.write_text(json.dumps({
                                "id": symbol.id,
                                "text": text,
                                "embedding": result,
                            }))
                            generated += 1
                    except Exception as e:
                        errors += 1
                        logger.debug(f"Error embedding {symbol.name}: {e}")
                    await asyncio.sleep(0.1)
            else:
                # Second: routines without embeddings
                missing_routines = [
                    r for r in self._kb._routines.values()
                    if f"routine:{r.name}" not in existing
                ]

                if missing_routines:
                    batch = missing_routines[:self.batch_size]
                    logger.info(f"Processing {len(batch)} routines...")

                    for routine in batch:
                        try:
                            text = routine.name
                            if routine.description:
                                text = f"{routine.name}: {routine.description}"
                            elif routine.calls:
                                text = f"{routine.name} calls {len(routine.calls)} subroutines"

                            result = await self._orchestrator.embed(text)
                            if result:
                                emb_id = f"routine:{routine.name}"
                                self._kb._embeddings[emb_id] = result
                                emb_file = self._kb.embeddings_dir / f"{hashlib.md5(emb_id.encode()).hexdigest()[:12]}.json"
                                emb_file.write_text(json.dumps({
                                    "id": emb_id,
                                    "text": text,
                                    "embedding": result,
                                }))
                                generated += 1
                        except Exception as e:
                            errors += 1
                            logger.debug(f"Error embedding {routine.name}: {e}")
                        await asyncio.sleep(0.1)

            if generated > 0:
                total_routines = len(self._kb._routines)
                routine_embs = sum(1 for k in self._kb._embeddings if k.startswith("routine:"))
                logger.info(f"Batch complete: {generated} generated, {errors} errors")
                logger.info(f"Embeddings: {len(self._kb._embeddings)} total, routines: {routine_embs}/{total_routines}")

            return generated

        except Exception as e:
            logger.error(f"Batch error: {e}")
            return 0

    async def run_once(self) -> int:
        """Run a single batch and update status."""
        generated = await self._run_batch()
        if generated:
            self._daily_count += generated
            self._record_progress()
        else:
            await self._maybe_trigger_post_completion()
        self._update_status(generated)
        return generated

    async def _sleep_until_midnight(self):
        """Sleep until midnight."""
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = tomorrow.replace(day=tomorrow.day + 1)
        sleep_seconds = (tomorrow - now).total_seconds()
        logger.info(f"Sleeping {sleep_seconds/3600:.1f} hours until midnight...")
        await asyncio.sleep(min(sleep_seconds, 3600))  # Check hourly

    def _update_status(self, generated: int):
        """Update daemon status file."""
        try:
            stats = self._kb.get_statistics() if self._kb else {}
            status = {
                "pid": os.getpid(),
                "running": self._running,
                "last_update": datetime.now().isoformat(),
                "daily_count": self._daily_count,
                "daily_limit": self.max_daily,
                "batch_size": self.batch_size,
                "interval_seconds": self.interval_seconds,
                "last_batch_generated": generated,
                "total_symbols": stats.get("total_symbols", 0),
                "total_embeddings": stats.get("total_embeddings", 0),
                "coverage_percent": round(
                    100 * stats.get("total_embeddings", 0) / max(stats.get("total_symbols", 1), 1),
                    1
                ),
            }
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("Cleaning up...")
        if self.pid_file.exists():
            self.pid_file.unlink()

        # Final status update
        try:
            status = {"running": False, "stopped": datetime.now().isoformat()}
            self.status_file.write_text(json.dumps(status, indent=2))
        except:
            pass

        logger.info("Daemon stopped")


def get_status() -> dict:
    """Get daemon status."""
    status_file = Path.home() / ".context" / "embedding_service" / "daemon_status.json"
    pid_file = Path.home() / ".context" / "embedding_service" / "daemon.pid"

    result = {"running": False}

    if status_file.exists():
        try:
            result = json.loads(status_file.read_text())
        except:
            pass

    # Check if process is actually running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            result["running"] = True
            result["pid"] = pid
        except (ProcessLookupError, ValueError):
            result["running"] = False

    return result


def install_launchd():
    """Install launchd plist for macOS."""
    python_path = resolve_python_executable()
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hafs.embedding-daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hafs.services.embedding_daemon</string>
        <string>--batch-size</string>
        <string>50</string>
        <string>--interval</string>
        <string>60</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{Path.home() / "Code" / "hafs"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{Path.home() / "Code" / "hafs" / "src"}</string>
        <key>HAFS_CONFIG_PATH</key>
        <string>{Path.home() / ".config" / "hafs" / "config.toml"}</string>
        <key>HAFS_PREFER_USER_CONFIG</key>
        <string>1</string>
        <key>GEMINI_API_KEY</key>
        <string>{os.environ.get("GEMINI_API_KEY", "")}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home() / ".context" / "logs" / "embedding_daemon.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{Path.home() / ".context" / "logs" / "embedding_daemon.err.log"}</string>
    <key>StartInterval</key>
    <integer>3600</integer>
</dict>
</plist>
"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.embedding-daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    print(f"Installed launchd plist: {plist_path}")
    print()
    print("To load the service:")
    print(f"  launchctl load {plist_path}")
    print()
    print("To unload:")
    print(f"  launchctl unload {plist_path}")
    print()
    print("To check status:")
    print("  launchctl list | grep hafs.embedding")


def uninstall_launchd():
    """Uninstall launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.embedding-daemon.plist"

    if plist_path.exists():
        os.system(f"launchctl unload {plist_path} 2>/dev/null")
        plist_path.unlink()
        print(f"Uninstalled: {plist_path}")
    else:
        print("Service not installed")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Embedding Daemon Service")
    parser.add_argument("--batch-size", "-b", type=int, default=50,
                        help="Embeddings per batch (default: 50)")
    parser.add_argument("--interval", "-i", type=int, default=60,
                        help="Seconds between batches (default: 60)")
    parser.add_argument("--max-daily", "-m", type=int, default=5000,
                        help="Maximum embeddings per day (default: 5000)")
    parser.add_argument("--install", action="store_true",
                        help="Install as launchd service")
    parser.add_argument("--uninstall", action="store_true",
                        help="Uninstall launchd service")
    parser.add_argument("--status", action="store_true",
                        help="Check daemon status")

    args = parser.parse_args()

    if args.install:
        install_launchd()
        return

    if args.uninstall:
        uninstall_launchd()
        return

    if args.status:
        status = get_status()
        print(json.dumps(status, indent=2))
        return

    # Run daemon
    daemon = EmbeddingDaemon(
        batch_size=args.batch_size,
        interval_seconds=args.interval,
        max_daily=args.max_daily,
    )
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
