"""Intelligent Quota Manager for rate limiting.

Features:
- Multi-process safe with file locking
- Exponential backoff on 429 errors
- Soft limits (warn at 80% usage)
- Dynamic limit loading from config/models.toml
- Request history tracking for analytics
- Smart health evaluation and recommendations
"""

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import QUOTA_USAGE_FILE

# Conditional imports for file locking
_PLATFORM = platform.system()
_HAS_FCNTL = False

if _PLATFORM != "Windows":
    try:
        import fcntl

        _HAS_FCNTL = True
    except ImportError:
        _HAS_FCNTL = False
else:
    try:
        import msvcrt
    except ImportError:
        msvcrt = None  # type: ignore


@dataclass
class UsageStats:
    """Usage statistics for a single model."""

    rpm: int = 0  # Requests per minute
    tpm: int = 0  # Tokens per minute
    rpd: int = 0  # Requests per day
    tpd: int = 0  # Tokens per day
    last_reset_min: float = 0
    last_reset_day: float = 0
    # Backoff state
    consecutive_429s: int = 0
    last_429_time: float = 0
    backoff_until: float = 0
    # History for analytics
    total_requests: int = 0
    total_tokens: int = 0
    total_429s: int = 0


@dataclass
class QuotaLimits:
    """Quota limits for a model."""

    tpm: int = 1_000_000  # Tokens per minute
    rpd: int = 10_000  # Requests per day
    rpm: int = 60  # Requests per minute (derived or explicit)
    soft_limit_pct: float = 0.80  # Warn at 80%


class QuotaManager:
    """Intelligent quota manager with exponential backoff and analytics.

    Features:
    - Exponential backoff: 1s, 2s, 4s, 8s... up to 5 min on 429s
    - Soft limits: Returns 'soft' availability at 80% usage
    - Dynamic limits: Loads from config/models.toml if available
    - Analytics: Tracks request history and success rates
    - Health status: Reports overall system health
    """

    # Default limits (fallback if config not available)
    # Model names should match core.models.registry
    DEFAULT_LIMITS = {
        # Gemini 3 series (current)
        "gemini-3-pro": QuotaLimits(tpm=2_000_000, rpd=2000, rpm=60),
        "gemini-3-flash": QuotaLimits(tpm=1_000_000, rpd=10_000, rpm=120),
        # Claude (current)
        "claude-opus-4.5": QuotaLimits(tpm=500_000, rpd=5_000, rpm=30),
        "claude-sonnet-4": QuotaLimits(tpm=500_000, rpd=5_000, rpm=60),
        "claude-haiku-3.5": QuotaLimits(tpm=1_000_000, rpd=10_000, rpm=120),
        # OpenAI GPT-5.2 (current)
        "gpt-5.2": QuotaLimits(tpm=500_000, rpd=10_000, rpm=60),
        "gpt-5.2-mini": QuotaLimits(tpm=1_000_000, rpd=50_000, rpm=120),
        # Local GPU (unlimited)
        "qwen-coder-14b": QuotaLimits(tpm=100_000_000, rpd=100_000, rpm=1000),
        "qwen-coder-32b": QuotaLimits(tpm=100_000_000, rpd=100_000, rpm=1000),
    }

    # Backoff configuration
    BACKOFF_BASE_SECONDS = 1.0
    BACKOFF_MAX_SECONDS = 300.0  # 5 minutes max
    BACKOFF_JITTER = 0.1  # 10% jitter

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or QUOTA_USAGE_FILE
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._limits_cache: Optional[Dict[str, QuotaLimits]] = None
        self._limits_cache_time: float = 0

    def _get_limits(self, model: str) -> QuotaLimits:
        """Get limits for a model, loading from config if available."""
        # Try to match model to known limits
        model_lower = model.lower()

        # Check exact match first
        for key, limits in self.DEFAULT_LIMITS.items():
            if key in model_lower:
                return limits

        # Try loading from config
        try:
            from core.model_config import get_model_config

            config = get_model_config()
            quota_data = config.get_quota_limits(model)
            if quota_data:
                return QuotaLimits(
                    tpm=quota_data.get("tpm", 1_000_000),
                    rpd=quota_data.get("rpd", 10_000),
                    rpm=quota_data.get("rpm", 60),
                )
        except Exception:
            pass

        # Default fallback
        return QuotaLimits()

    def _load_state(self, file_handle) -> Dict[str, UsageStats]:
        try:
            file_handle.seek(0)
            content = file_handle.read()
            if not content:
                return {}
            data = json.loads(content)
            return {k: UsageStats(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_state(self, file_handle, usage_data: Dict[str, UsageStats]):
        file_handle.seek(0)
        file_handle.truncate()
        json.dump({k: asdict(v) for k, v in usage_data.items()}, file_handle)
        file_handle.flush()

    def _with_lock(self, operation):
        """Execute an operation with an exclusive file lock."""
        if not self.state_path.exists():
            self.state_path.write_text("{}")

        with open(self.state_path, "r+") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    usage = self._load_state(f)
                    result = operation(usage)
                    self._save_state(f, usage)
                    return result
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            elif _PLATFORM == "Windows" and msvcrt is not None:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        usage = self._load_state(f)
                        result = operation(usage)
                        self._save_state(f, usage)
                        return result
                    finally:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    usage = self._load_state(f)
                    result = operation(usage)
                    self._save_state(f, usage)
                    return result
            else:
                usage = self._load_state(f)
                result = operation(usage)
                self._save_state(f, usage)
                return result

    def _reset_counters(self, usage: Dict[str, UsageStats], model: str):
        """Reset counters if time windows have elapsed."""
        if model not in usage:
            usage[model] = UsageStats()

        stats = usage[model]
        now = time.time()

        # Reset per-minute counters
        if now - stats.last_reset_min > 60:
            stats.rpm = 0
            stats.tpm = 0
            stats.last_reset_min = now

        # Reset per-day counters
        if now - stats.last_reset_day > 86400:
            stats.rpd = 0
            stats.tpd = 0
            stats.last_reset_day = now

    def check_availability(
        self, model: str, estimated_tokens: int = 1000
    ) -> str:
        """Check if a request can be made.

        Returns:
            'available': Request can proceed
            'soft_limit': Approaching limits (>80%), proceed with caution
            'backoff': Currently in backoff period, wait
            'rate_limited': Would exceed limits, use different model
        """

        def _check(usage: Dict[str, UsageStats]) -> str:
            self._reset_counters(usage, model)
            stats = usage[model]
            limits = self._get_limits(model)
            now = time.time()

            # Check if in backoff period
            if stats.backoff_until > now:
                return "backoff"

            # Check hard limits
            if stats.tpm + estimated_tokens > limits.tpm:
                return "rate_limited"
            if stats.rpd + 1 > limits.rpd:
                return "rate_limited"
            if stats.rpm + 1 > limits.rpm:
                return "rate_limited"

            # Check soft limits (80%)
            soft_tpm = limits.tpm * limits.soft_limit_pct
            soft_rpd = limits.rpd * limits.soft_limit_pct
            if stats.tpm + estimated_tokens > soft_tpm:
                return "soft_limit"
            if stats.rpd + 1 > soft_rpd:
                return "soft_limit"

            return "available"

        return self._with_lock(_check)

    def log_usage(self, model: str, tokens: int, success: bool = True):
        """Log a request's usage."""

        def _log(usage: Dict[str, UsageStats]):
            self._reset_counters(usage, model)
            stats = usage[model]

            stats.rpm += 1
            stats.rpd += 1
            stats.tpm += tokens
            stats.tpd += tokens
            stats.total_requests += 1
            stats.total_tokens += tokens

            if success:
                # Reset backoff on success
                stats.consecutive_429s = 0
                stats.backoff_until = 0

        self._with_lock(_log)

    def log_rate_limit_error(self, model: str):
        """Log a 429 error and apply exponential backoff."""

        def _log_429(usage: Dict[str, UsageStats]):
            self._reset_counters(usage, model)
            stats = usage[model]
            now = time.time()

            stats.consecutive_429s += 1
            stats.total_429s += 1
            stats.last_429_time = now

            # Calculate backoff with exponential increase and jitter
            backoff_seconds = min(
                self.BACKOFF_BASE_SECONDS * (2 ** (stats.consecutive_429s - 1)),
                self.BACKOFF_MAX_SECONDS,
            )
            # Add jitter
            jitter = backoff_seconds * self.BACKOFF_JITTER * random.random()
            backoff_seconds += jitter

            stats.backoff_until = now + backoff_seconds
            return backoff_seconds

        return self._with_lock(_log_429)

    def get_backoff_time(self, model: str) -> float:
        """Get remaining backoff time in seconds."""

        def _get_backoff(usage: Dict[str, UsageStats]) -> float:
            if model not in usage:
                return 0
            stats = usage[model]
            remaining = stats.backoff_until - time.time()
            return max(0, remaining)

        return self._with_lock(_get_backoff)

    def get_usage_stats(self, model: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a model."""

        def _get_stats(usage: Dict[str, UsageStats]) -> Optional[Dict[str, Any]]:
            if model not in usage:
                return None
            stats = usage[model]
            limits = self._get_limits(model)
            return {
                "rpm": stats.rpm,
                "tpm": stats.tpm,
                "rpd": stats.rpd,
                "tpd": stats.tpd,
                "rpm_limit": limits.rpm,
                "tpm_limit": limits.tpm,
                "rpd_limit": limits.rpd,
                "usage_pct_tpm": (stats.tpm / limits.tpm * 100) if limits.tpm else 0,
                "usage_pct_rpd": (stats.rpd / limits.rpd * 100) if limits.rpd else 0,
                "total_requests": stats.total_requests,
                "total_tokens": stats.total_tokens,
                "total_429s": stats.total_429s,
                "success_rate": (
                    (stats.total_requests - stats.total_429s) / stats.total_requests * 100
                    if stats.total_requests > 0
                    else 100
                ),
                "in_backoff": stats.backoff_until > time.time(),
                "backoff_remaining": max(0, stats.backoff_until - time.time()),
            }

        return self._with_lock(_get_stats)

    def evaluate_system_health(self) -> str:
        """Evaluate overall system health across all models.

        Returns:
            'HEALTHY': All models available, low usage
            'DEGRADED': Some models in soft limit or backoff
            'CRITICAL': Most models rate limited or in backoff
        """

        def _evaluate(usage: Dict[str, UsageStats]) -> str:
            if not usage:
                return "HEALTHY"

            now = time.time()
            healthy = 0
            degraded = 0
            critical = 0

            for model, stats in usage.items():
                self._reset_counters(usage, model)
                limits = self._get_limits(model)

                # Check backoff
                if stats.backoff_until > now:
                    critical += 1
                    continue

                # Check usage levels
                tpm_pct = stats.tpm / limits.tpm if limits.tpm else 0
                rpd_pct = stats.rpd / limits.rpd if limits.rpd else 0

                if tpm_pct > 0.9 or rpd_pct > 0.9:
                    critical += 1
                elif tpm_pct > 0.7 or rpd_pct > 0.7:
                    degraded += 1
                else:
                    healthy += 1

            total = healthy + degraded + critical
            if total == 0:
                return "HEALTHY"

            if critical / total > 0.5:
                return "CRITICAL"
            elif degraded / total > 0.3:
                return "DEGRADED"
            return "HEALTHY"

        return self._with_lock(_evaluate)

    def recommend_scale(self) -> str:
        """Recommend scaling action based on usage patterns.

        Returns:
            'SCALE_DOWN': Very low usage, can reduce parallelism
            'MEDIUM': Normal usage, maintain current scale
            'SCALE_UP': High usage with headroom, can increase
            'THROTTLE': Near limits, should reduce request rate
        """

        def _recommend(usage: Dict[str, UsageStats]) -> str:
            if not usage:
                return "MEDIUM"

            total_tpm_usage = 0
            total_tpm_limit = 0

            for model, stats in usage.items():
                limits = self._get_limits(model)
                total_tpm_usage += stats.tpm
                total_tpm_limit += limits.tpm

            if total_tpm_limit == 0:
                return "MEDIUM"

            usage_pct = total_tpm_usage / total_tpm_limit

            if usage_pct < 0.2:
                return "SCALE_DOWN"
            elif usage_pct < 0.5:
                return "MEDIUM"
            elif usage_pct < 0.8:
                return "SCALE_UP"
            else:
                return "THROTTLE"

        return self._with_lock(_recommend)

    def get_best_available_model(self, models: list[str]) -> Optional[str]:
        """Get the best available model from a list (lowest usage)."""

        def _find_best(usage: Dict[str, UsageStats]) -> Optional[str]:
            best_model = None
            best_score = float("inf")

            for model in models:
                self._reset_counters(usage, model)
                stats = usage[model]
                limits = self._get_limits(model)

                # Skip if in backoff
                if stats.backoff_until > time.time():
                    continue

                # Skip if at limits
                if stats.tpm >= limits.tpm or stats.rpd >= limits.rpd:
                    continue

                # Score based on usage percentage (lower is better)
                tpm_pct = stats.tpm / limits.tpm if limits.tpm else 0
                rpd_pct = stats.rpd / limits.rpd if limits.rpd else 0
                score = (tpm_pct + rpd_pct) / 2

                if score < best_score:
                    best_score = score
                    best_model = model

            return best_model

        return self._with_lock(_find_best)


# Global singleton
quota_manager = QuotaManager()
