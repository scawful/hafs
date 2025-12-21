#!/usr/bin/env python3
"""Hybrid GPU + API Orchestrator for Training Data Generation.

Intelligently routes generation requests between:
1. Local GPU (medical-mechanica 5060TI) - FREE but limited capacity
2. Gemini API - PAID but unlimited capacity

Load balancing strategy:
- GPU utilization < 70%: Route to GPU (free)
- GPU utilization 70-90%: Distribute 50/50 between GPU and API
- GPU utilization > 90%: Route to API (prevent overload)

This maximizes GPU usage while preventing overload and maintaining throughput.
"""

from __future__ import annotations

import asyncio
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUStatus:
    """GPU resource status."""

    utilization_percent: float  # GPU compute utilization
    memory_used_mb: int  # GPU memory used
    memory_total_mb: int  # GPU memory total
    temperature_c: int  # GPU temperature
    power_watts: int  # Power draw
    timestamp: datetime
    is_healthy: bool  # Overall health status


class GPUMonitor:
    """Monitor medical-mechanica GPU via SSH."""

    def __init__(self, host: str = "medical-mechanica"):
        self.host = host
        self._cache: Optional[GPUStatus] = None
        self._cache_ttl = timedelta(seconds=5)  # Cache for 5 seconds
        self._last_check: Optional[datetime] = None

    async def get_status(self) -> Optional[GPUStatus]:
        """Get current GPU status (cached)."""
        now = datetime.now()

        # Return cached if fresh
        if (
            self._cache
            and self._last_check
            and (now - self._last_check) < self._cache_ttl
        ):
            return self._cache

        # Query GPU
        try:
            # Run nvidia-smi via SSH
            cmd = [
                "ssh",
                self.host,
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {stderr.decode()}")
                return None

            # Parse output: "85, 12345, 16384, 72, 180"
            line = stdout.decode().strip()
            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 5:
                logger.warning(f"Unexpected nvidia-smi output: {line}")
                return None

            status = GPUStatus(
                utilization_percent=float(parts[0]),
                memory_used_mb=int(float(parts[1])),
                memory_total_mb=int(float(parts[2])),
                temperature_c=int(float(parts[3])),
                power_watts=int(float(parts[4])),
                timestamp=now,
                is_healthy=True,
            )

            # Health checks
            if status.temperature_c > 85:
                logger.warning(f"GPU temperature high: {status.temperature_c}°C")
                status.is_healthy = False

            if status.memory_used_mb > status.memory_total_mb * 0.95:
                logger.warning("GPU memory near capacity")
                status.is_healthy = False

            self._cache = status
            self._last_check = now

            return status

        except Exception as e:
            logger.error(f"Error checking GPU status: {e}")
            return None


class HybridLoadBalancer:
    """Route generation requests between GPU and API based on load."""

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        gpu_threshold_low: float = 70.0,  # Below this: prefer GPU
        gpu_threshold_high: float = 90.0,  # Above this: prefer API
    ):
        self.gpu_monitor = gpu_monitor
        self.gpu_threshold_low = gpu_threshold_low
        self.gpu_threshold_high = gpu_threshold_high

        # Routing stats
        self.gpu_requests = 0
        self.api_requests = 0
        self.gpu_failures = 0

    async def should_use_gpu(self) -> tuple[bool, str]:
        """Decide whether to use GPU or API.

        Returns:
            (use_gpu: bool, reason: str)
        """
        status = await self.gpu_monitor.get_status()

        # GPU unavailable
        if not status:
            return False, "GPU status unavailable"

        if not status.is_healthy:
            return False, f"GPU unhealthy (temp={status.temperature_c}°C)"

        util = status.utilization_percent

        # GPU underutilized - USE IT!
        if util < self.gpu_threshold_low:
            return True, f"GPU available ({util:.0f}% util)"

        # GPU heavily loaded - avoid
        if util > self.gpu_threshold_high:
            return False, f"GPU overloaded ({util:.0f}% util)"

        # GPU moderately loaded - probabilistic routing
        # Linear interpolation: 70% util = 50% GPU, 80% util = 25% GPU, 90% util = 0% GPU
        gpu_probability = 1.0 - (
            (util - self.gpu_threshold_low)
            / (self.gpu_threshold_high - self.gpu_threshold_low)
        )
        use_gpu = random.random() < gpu_probability

        reason = f"GPU {util:.0f}% util, {gpu_probability*100:.0f}% prob"
        return use_gpu, reason

    def record_request(self, used_gpu: bool, success: bool = True):
        """Record routing decision for stats."""
        if used_gpu:
            self.gpu_requests += 1
            if not success:
                self.gpu_failures += 1
        else:
            self.api_requests += 1

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = self.gpu_requests + self.api_requests
        return {
            "total_requests": total,
            "gpu_requests": self.gpu_requests,
            "api_requests": self.api_requests,
            "gpu_percentage": (
                self.gpu_requests / total * 100 if total > 0 else 0
            ),
            "gpu_failures": self.gpu_failures,
        }


async def test_hybrid_system():
    """Test the hybrid GPU + API system."""
    print("Testing Hybrid GPU + API System")
    print("=" * 60)

    # Initialize monitor
    monitor = GPUMonitor()
    balancer = HybridLoadBalancer(monitor)

    # Check GPU status
    print("\n[1] Checking GPU status...")
    status = await monitor.get_status()

    if status:
        print(f"✓ GPU Online")
        print(f"  Utilization: {status.utilization_percent:.1f}%")
        print(f"  Memory: {status.memory_used_mb}MB / {status.memory_total_mb}MB")
        print(f"  Temperature: {status.temperature_c}°C")
        print(f"  Power: {status.power_watts}W")
        print(f"  Healthy: {status.is_healthy}")
    else:
        print("✗ GPU Unavailable")

    # Test routing decisions
    print("\n[2] Testing routing decisions...")
    for i in range(10):
        use_gpu, reason = await balancer.should_use_gpu()
        target = "GPU (FREE)" if use_gpu else "API (PAID)"
        print(f"  Request {i+1}: {target} - {reason}")
        balancer.record_request(use_gpu)
        await asyncio.sleep(0.1)

    # Show stats
    print("\n[3] Routing statistics:")
    stats = balancer.get_stats()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  GPU: {stats['gpu_requests']} ({stats['gpu_percentage']:.1f}%)")
    print(f"  API: {stats['api_requests']} ({100-stats['gpu_percentage']:.1f}%)")

    print("\n✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_hybrid_system())
