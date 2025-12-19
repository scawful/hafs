import time
import json
import fcntl
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from hafs.core.config import QUOTA_USAGE_FILE

@dataclass
class UsageStats:
    rpm: int = 0
    tpm: int = 0
    rpd: int = 0
    last_reset_min: float = 0
    last_reset_day: float = 0

class QuotaManager:
    """Tracks estimated usage to avoid 429s (Multi-Process Safe)."""
    
    LIMITS = {
        # Reasoning
        "gemini-3-pro": {"tpm": 1_000_000, "rpd": 250},
        "gemini-2.5-pro": {"tpm": 2_000_000, "rpd": 10_000},
        "gemini-1.5-pro": {"tpm": 2_000_000, "rpd": 10_000},
        
        # Fast
        "gemini-3-flash": {"tpm": 1_000_000, "rpd": 10_000},
        "gemini-2.5-flash": {"tpm": 1_000_000, "rpd": 10_000},
        "gemini-1.5-flash": {"tpm": 4_000_000, "rpd": 999_999},
        "gemini-2.0-flash": {"tpm": 4_000_000, "rpd": 999_999},
    }

    def __init__(self):
        self.state_path = QUOTA_USAGE_FILE
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # We don't load state in __init__ anymore, we load it on demand under lock

    def _load_state(self, file_handle) -> Dict[str, UsageStats]:
        try:
            file_handle.seek(0)
            content = file_handle.read()
            if not content: return {}
            data = json.loads(content)
            return {k: UsageStats(**v) for k, v in data.items()}
        except:
            return {}

    def _save_state(self, file_handle, usage_data):
        file_handle.seek(0)
        file_handle.truncate()
        json.dump({k: asdict(v) for k, v in usage_data.items()}, file_handle)
        file_handle.flush()

    def _with_lock(self, operation):
        """Execute an operation with an exclusive file lock."""
        # Ensure file exists
        if not self.state_path.exists():
            self.state_path.write_text("{}")
            
        with open(self.state_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX) # Exclusive lock (blocking)
            try:
                usage = self._load_state(f)
                result = operation(usage)
                self._save_state(f, usage)
                return result
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def check_availability(self, model: str, estimated_tokens: int = 1000) -> bool:
        def _check(usage):
            self._reset_counters(usage, model)
            stats = usage[model]
            
            # Map model to limits
            limit_key = model
            for k in self.LIMITS.keys():
                if k in model: 
                    limit_key = k
                    break
            
            limits = self.LIMITS.get(limit_key)
            if not limits: return True # No limits defined
                
            if stats.tpm + estimated_tokens > limits["tpm"]: return False
            if stats.rpd + 1 > limits["rpd"]: return False
            return True

        return self._with_lock(_check)

    def log_usage(self, model: str, tokens: int):
        def _log(usage):
            self._reset_counters(usage, model)
            stats = usage[model]
            stats.rpm += 1
            stats.rpd += 1
            stats.tpm += tokens
            
        self._with_lock(_log)

    def _reset_counters(self, usage, model: str):
        if model not in usage:
            usage[model] = UsageStats()
            
        stats = usage[model]
        now = time.time()
        
        if now - stats.last_reset_min > 60:
            stats.rpm = 0
            stats.tpm = 0
            stats.last_reset_min = now
            
        if now - stats.last_reset_day > 86400:
            stats.rpd = 0
            stats.last_reset_day = now

    def evaluate_system_health(self) -> str:
        return "HEALTHY"

    def recommend_scale(self) -> str:
        return "MEDIUM"

quota_manager = QuotaManager()
