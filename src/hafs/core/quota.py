import time
import json
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
    """Tracks estimated usage to avoid 429s."""
    
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
        self.usage: Dict[str, UsageStats] = self._load_state()

    def _load_state(self) -> Dict[str, UsageStats]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text())
            return {k: UsageStats(**v) for k, v in data.items()}
        except:
            return {}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.usage.items()}, f)

    def _reset_counters(self, model: str):
        if model not in self.usage:
            self.usage[model] = UsageStats()
            
        stats = self.usage[model]
        now = time.time()
        
        if now - stats.last_reset_min > 60:
            stats.rpm = 0
            stats.tpm = 0
            stats.last_reset_min = now
            
        if now - stats.last_reset_day > 86400:
            stats.rpd = 0
            stats.last_reset_day = now

    def check_availability(self, model: str, estimated_tokens: int = 1000) -> bool:
        self._reset_counters(model)
        
        limit_key = model
        if "gemini-3-pro" in model: limit_key = "gemini-3-pro"
        elif "gemini-3-flash" in model: limit_key = "gemini-3-flash"
        elif "gemini-2.5-pro" in model: limit_key = "gemini-2.5-pro"
        elif "gemini-2.5-flash" in model: limit_key = "gemini-2.5-flash"
        elif "gemini-1.5-pro" in model: limit_key = "gemini-1.5-pro"
        elif "gemini-1.5-flash" in model: limit_key = "gemini-1.5-flash"
        
        limits = self.LIMITS.get(limit_key)
        if not limits: return True
            
        stats = self.usage[model]
        
        if stats.tpm + estimated_tokens > limits["tpm"]: return False
        if stats.rpd + 1 > limits["rpd"]: return False
            
        return True

    def log_usage(self, model: str, tokens: int):
        self._reset_counters(model)
        stats = self.usage[model]
        stats.rpm += 1
        stats.rpd += 1
        stats.tpm += tokens
        self._save_state()

    def evaluate_system_health(self) -> str:
        """Simple health check based on daily limits."""
        # For now, simplistic
        return "HEALTHY"

    def recommend_scale(self) -> str:
        """Recommends MEDIUM by default, could be more intelligent."""
        return "MEDIUM"

quota_manager = QuotaManager()
