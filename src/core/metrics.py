import json
from datetime import datetime
from pathlib import Path
from core.config import hafs_config

class MetricsLogger:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics_dir = hafs_config.context_root / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.metrics_dir / "agents.jsonl"

    def log_start(self):
        self._write({"agent": self.agent_name, "event": "start", "timestamp": datetime.now().isoformat()})

    def log_success(self, items_processed: int = 0):
        self._write({"agent": self.agent_name, "event": "success", "items_processed": items_processed, "timestamp": datetime.now().isoformat()})

    def log_failure(self, error: str):
        self._write({"agent": self.agent_name, "event": "failure", "error": error, "timestamp": datetime.now().isoformat()})

    def _write(self, data: dict):
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except: pass
