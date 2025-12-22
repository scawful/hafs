#!/usr/bin/env python3
"""Generate Oracle-Farore dataset with hybrid GPU + multi-provider API.

Uses centralized model registry for all model configurations.
Supports resume from checkpoint and robust error handling.

Providers (from registry):
- GPU: qwen-coder-14b (FREE local inference)
- Gemini: gemini-3-flash (fast, cheap)
- Anthropic: claude-opus-4.5 (highest quality)
- OpenAI: gpt-5.2 (excellent for code)

Intelligent routing:
- GPU <70% load → Use GPU (free)
- GPU 70-90% → Use Gemini (fast/cheap)
- Complex Oracle samples → Route to Opus 4.5 (best quality)
- ASM samples → Mix Gemini + GPT-5.2
- Gigaleak → Use GPT-5.2 (good at code comparison)

Features:
- Resume from checkpoint on crash/restart
- Exponential backoff on failures
- Provider health tracking
- Cost estimation

Target: 1000+ samples, <$2 API cost
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models.registry import get_model, get_model_id, MODELS

from agents.training.curator import DataCurator
from agents.training.generators.oracle_generator import OracleDataGenerator
from agents.training.generators.asm_generator import AsmDataGenerator
from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
from agents.training.hybrid_orchestrator import GPUMonitor, HybridLoadBalancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Checkpoint directory
CHECKPOINT_DIR = Path.home() / ".context" / "training" / "checkpoints"


@dataclass
class GenerationCheckpoint:
    """Checkpoint for resuming generation."""
    output_name: str
    started_at: str
    last_updated: str
    target_count: int
    completed_count: int
    domains_progress: dict = field(default_factory=dict)
    provider_usage: dict = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    failed_items: list = field(default_factory=list)

    def save(self):
        """Save checkpoint to disk."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_DIR / f"{self.output_name}.json"
        self.last_updated = datetime.now().isoformat()
        with open(checkpoint_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    @classmethod
    def load(cls, output_name: str) -> Optional["GenerationCheckpoint"]:
        """Load checkpoint from disk."""
        checkpoint_path = CHECKPOINT_DIR / f"{output_name}.json"
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    @classmethod
    def create(cls, output_name: str, target_count: int) -> "GenerationCheckpoint":
        """Create new checkpoint."""
        return cls(
            output_name=output_name,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            target_count=target_count,
            completed_count=0,
        )


@dataclass
class ProviderHealth:
    """Track provider health for intelligent routing."""
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0

    def record_success(self):
        self.consecutive_failures = 0
        self.total_requests += 1

    def record_failure(self):
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = time.time()

    def is_healthy(self) -> bool:
        """Check if provider is healthy (not in cooldown)."""
        if self.consecutive_failures >= 3:
            # Exponential backoff: 30s, 60s, 120s, etc.
            cooldown = min(30 * (2 ** (self.consecutive_failures - 3)), 300)
            if time.time() - self.last_failure_time < cooldown:
                return False
        return True

    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests


class MultiProviderOrchestrator:
    """Orchestrator that routes between GPU and multiple API providers.

    Uses centralized model registry for all model configurations.
    Includes health tracking, retry logic, and cost estimation.
    """

    # Model names from registry
    GPU_MODEL = "qwen-coder-14b"
    GEMINI_MODEL = "gemini-3-flash"
    OPUS_MODEL = "claude-opus-4.5"
    GPT_MODEL = "gpt-5.2"

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        load_balancer: HybridLoadBalancer,
        max_retries: int = 3,
    ):
        self.gpu_monitor = gpu_monitor
        self.load_balancer = load_balancer
        self.max_retries = max_retries
        self._gpu_backend = None
        self._providers = {}
        self._request_count = {"gpu": 0, "gemini": 0, "opus": 0, "gpt": 0}
        self._provider_health = {
            "gpu": ProviderHealth(),
            "gemini": ProviderHealth(),
            "opus": ProviderHealth(),
            "gpt": ProviderHealth(),
        }
        self._estimated_cost = 0.0

    async def setup(self):
        """Initialize GPU backend and all API providers."""
        import os
        import subprocess

        # GPU backend (Ollama on medical-mechanica)
        gpu_config = get_model(self.GPU_MODEL)
        try:
            from backends.api.ollama import OllamaBackend
            self._gpu_backend = OllamaBackend(
                host="100.104.53.21",
                port=11434,
                model=gpu_config.model_id,
                timeout=60.0,
            )
            logger.info(f"✓ GPU backend: {gpu_config.display_name} (Ollama)")
        except Exception as e:
            logger.warning(f"GPU backend failed: {e}")
            self._gpu_backend = None

        # Gemini - only if API key configured
        gemini_config = get_model(self.GEMINI_MODEL)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self._providers["gemini"] = genai.GenerativeModel(gemini_config.model_id)
                logger.info(f"✓ {gemini_config.display_name}")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
        else:
            logger.info("⊘ Gemini: GEMINI_API_KEY not set")

        # Claude Opus - API key first, then CLI fallback
        opus_config = get_model(self.OPUS_MODEL)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.startswith("sk-ant-api"):
            try:
                from anthropic import Anthropic
                self._providers["opus"] = Anthropic(api_key=anthropic_key)
                self._opus_mode = "api"
                logger.info(f"✓ {opus_config.display_name} (Console API)")
            except Exception as e:
                logger.warning(f"Anthropic SDK init failed: {e}")
        else:
            # Try Claude Code CLI as fallback
            try:
                result = subprocess.run(
                    ["claude", "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self._providers["opus"] = "claude-cli"
                    self._opus_mode = "cli"
                    logger.info(f"✓ {opus_config.display_name} (Claude Code CLI - Max subscription)")
                else:
                    logger.info("⊘ Opus: No API key and CLI unavailable")
            except FileNotFoundError:
                logger.info("⊘ Opus: No API key and CLI not found")
            except Exception as e:
                logger.info(f"⊘ Opus: {e}")

        # GPT - only if API key configured
        gpt_config = get_model(self.GPT_MODEL)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self._providers["gpt"] = OpenAI(api_key=openai_key)
                logger.info(f"✓ {gpt_config.display_name}")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")
        else:
            logger.info(f"⊘ {gpt_config.display_name}: OPENAI_API_KEY not set")

    def _select_provider(self, domain: str, complexity: str = "medium") -> str:
        """Select best provider based on domain and complexity."""
        # Complex Oracle samples → Best flagship (Opus if available, else GPT-5.2)
        if domain == "oracle" and complexity == "high":
            if "opus" in self._providers:
                return "opus"
            if "gpt" in self._providers:
                return "gpt"  # GPT-5.2 as flagship fallback

        # Oracle medium complexity → Mix Gemini + GPT-5.2 (or Opus if available)
        if domain == "oracle":
            # 60% Gemini (cheap), 40% GPT-5.2 (quality)
            if "gpt" in self._providers and self._request_count["gemini"] > self._request_count["gpt"] * 1.5:
                return "gpt"
            if "opus" in self._providers and self._request_count["gemini"] > self._request_count["opus"] * 2.3:
                return "opus"
            if "gemini" in self._providers:
                return "gemini"

        # ASM samples → Mix Gemini + GPT
        if domain == "asm":
            # 60% Gemini, 40% GPT
            if "gpt" in self._providers and self._request_count["gemini"] > self._request_count["gpt"] * 1.5:
                return "gpt"
            if "gemini" in self._providers:
                return "gemini"

        # Gigaleak → Prefer GPT-5.2 (good at code comparison)
        if domain == "gigaleak":
            if "gpt" in self._providers:
                return "gpt"

        # Default: Gemini (fast/cheap)
        return "gemini"

    def _estimate_cost(self, provider_name: str, input_tokens: int = 1000, output_tokens: int = 500):
        """Estimate cost for a request."""
        model_map = {
            "gpu": self.GPU_MODEL,
            "gemini": self.GEMINI_MODEL,
            "opus": self.OPUS_MODEL,
            "gpt": self.GPT_MODEL,
        }
        model_name = model_map.get(provider_name)
        if not model_name:
            return 0.0
        config = get_model(model_name)
        input_cost = (input_tokens / 1_000_000) * config.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * config.cost_per_1m_output
        return input_cost + output_cost

    async def generate(self, prompt: str, domain: str = "unknown", **kwargs) -> "Response":
        """Generate with intelligent routing and retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            # Try GPU first if available and healthy
            use_gpu, reason = await self.load_balancer.should_use_gpu()

            if use_gpu and self._gpu_backend and self._provider_health["gpu"].is_healthy():
                try:
                    response = await self._gpu_backend.generate_one_shot(prompt)
                    self.load_balancer.record_request(used_gpu=True, success=True)
                    self._provider_health["gpu"].record_success()
                    self._request_count["gpu"] += 1
                    return Response(content=response, provider="gpu")
                except Exception as e:
                    logger.warning(f"GPU failed (attempt {attempt + 1}): {e}")
                    self.load_balancer.record_request(used_gpu=True, success=False)
                    self._provider_health["gpu"].record_failure()
                    last_error = e

            # Select best API provider considering health
            provider_name = self._select_provider(domain)

            # Skip unhealthy providers
            while provider_name and not self._provider_health[provider_name].is_healthy():
                logger.debug(f"Skipping unhealthy provider: {provider_name}")
                # Try next provider
                available = [p for p in self._providers.keys()
                           if p != provider_name and self._provider_health[p].is_healthy()]
                provider_name = available[0] if available else None

            provider = self._providers.get(provider_name) if provider_name else None

            if not provider:
                # Fallback to any available provider
                for name in self._providers.keys():
                    if self._provider_health[name].is_healthy():
                        provider_name = name
                        provider = self._providers[name]
                        break

            if not provider:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"No healthy providers, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError("No providers available after retries")

            # Generate based on provider type
            try:
                content = await self._generate_with_provider(provider_name, provider, prompt)
                self.load_balancer.record_request(used_gpu=False, success=True)
                self._provider_health[provider_name].record_success()
                self._request_count[provider_name] += 1

                # Estimate and track cost
                cost = self._estimate_cost(provider_name)
                self._estimated_cost += cost

                return Response(content=content, provider=provider_name)

            except Exception as e:
                logger.warning(f"{provider_name} failed (attempt {attempt + 1}): {e}")
                self._provider_health[provider_name].record_failure()
                last_error = e

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

        raise RuntimeError(f"All providers failed after {self.max_retries} attempts: {last_error}")

    async def _generate_with_provider(self, provider_name: str, provider: Any, prompt: str) -> str:
        """Generate content using a specific provider."""
        import subprocess

        if provider_name == "gemini":
            response = provider.generate_content(prompt)
            return response.text

        elif provider_name == "opus":
            opus_config = get_model(self.OPUS_MODEL)
            if getattr(self, '_opus_mode', 'cli') == "api":
                # Use Anthropic SDK
                response = provider.messages.create(
                    model=opus_config.model_id,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            else:
                # Use Claude Code CLI for Max subscription
                result = subprocess.run(
                    ["claude", "-p", "--model", "opus", "--output-format", "json", prompt],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Claude CLI failed: {result.stderr}")
                data = json.loads(result.stdout)
                return data.get("result", data.get("content", result.stdout))

        elif provider_name == "gpt":
            gpt_config = get_model(self.GPT_MODEL)
            response = provider.chat.completions.create(
                model=gpt_config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "request_counts": self._request_count.copy(),
            "estimated_cost_usd": self._estimated_cost,
            "provider_health": {
                name: {
                    "success_rate": health.success_rate(),
                    "total_requests": health.total_requests,
                    "consecutive_failures": health.consecutive_failures,
                }
                for name, health in self._provider_health.items()
            },
        }


class Response:
    """Response wrapper."""
    def __init__(self, content: str, provider: str = "unknown"):
        self.content = content
        self.provider = provider


async def main(resume_name: Optional[str] = None, target_count: int = 1000):
    """Generate Oracle dataset with hybrid GPU + multi-provider.

    Args:
        resume_name: Optional checkpoint name to resume from
        target_count: Target number of samples to generate
    """
    # Check for resume
    checkpoint = None
    if resume_name:
        checkpoint = GenerationCheckpoint.load(resume_name)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_name}")
            logger.info(f"  Progress: {checkpoint.completed_count}/{checkpoint.target_count}")
            logger.info(f"  Started: {checkpoint.started_at}")
            output_name = checkpoint.output_name
        else:
            logger.warning(f"Checkpoint not found: {resume_name}, starting fresh")

    logger.info("=" * 80)
    logger.info("ORACLE-FARORE HYBRID GENERATION")
    logger.info("=" * 80)
    logger.info(f"Target: {target_count} samples")

    # Show model info from registry
    for model_name in ["qwen-coder-14b", "gemini-3-flash", "claude-opus-4.5", "gpt-5.2"]:
        config = get_model(model_name)
        cost_info = f"${config.cost_per_1m_input:.2f}/${config.cost_per_1m_output:.2f}" if config.cost_per_1m_input > 0 else "FREE"
        logger.info(f"  {config.display_name}: {cost_info}")

    logger.info("\nRouting:")
    logger.info("  - GPU <70% → Free GPU inference")
    logger.info("  - Oracle complex → Opus 4.5 (best quality)")
    logger.info("  - Oracle medium → 70% Gemini, 30% Opus")
    logger.info("  - ASM → 60% Gemini, 40% GPT-5.2")
    logger.info("  - Gigaleak → GPT-5.2 (code comparison)")
    logger.info("=" * 80)

    # Initialize hybrid system
    logger.info("\nInitializing hybrid GPU + multi-API system...")
    gpu_monitor = GPUMonitor()
    load_balancer = HybridLoadBalancer(
        gpu_monitor, gpu_threshold_low=70.0, gpu_threshold_high=90.0
    )

    # Check GPU
    gpu_status = await gpu_monitor.get_status()
    if gpu_status:
        logger.info(f"✓ GPU Online: {gpu_status.utilization_percent:.0f}% utilized")
        logger.info(f"  Memory: {gpu_status.memory_used_mb}MB / {gpu_status.memory_total_mb}MB")
    else:
        logger.warning("⚠️  GPU Unavailable - API only mode")

    # Initialize multi-provider orchestrator
    hybrid_orch = MultiProviderOrchestrator(gpu_monitor, load_balancer)
    await hybrid_orch.setup()

    # Restore provider usage from checkpoint
    if checkpoint and checkpoint.provider_usage:
        hybrid_orch._request_count.update(checkpoint.provider_usage)
        hybrid_orch._estimated_cost = checkpoint.estimated_cost_usd

    # Initialize curator
    logger.info("\nInitializing DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register Oracle generator (primary)
    logger.info("\nRegistering generators with hybrid routing...")
    oracle_gen = OracleDataGenerator(use_enhanced_prompts=True)
    await oracle_gen.setup()
    # Patch orchestrator
    if hasattr(oracle_gen, "_orchestrator") and oracle_gen._orchestrator:
        oracle_gen._orchestrator.generate = lambda prompt, **kw: hybrid_orch.generate(prompt, domain="oracle", **kw)
    curator.register_generator("oracle", oracle_gen)
    logger.info("✓ Oracle (Opus 4.5 + Gemini mix)")

    # Register ASM generator
    asm_gen = AsmDataGenerator(use_enhanced_prompts=True)
    await asm_gen.setup()
    if hasattr(asm_gen, "_orchestrator") and asm_gen._orchestrator:
        asm_gen._orchestrator.generate = lambda prompt, **kw: hybrid_orch.generate(prompt, domain="asm", **kw)
    curator.register_generator("asm", asm_gen)
    logger.info("✓ ASM (Gemini + GPT-5.2 mix)")

    # Register Gigaleak generator
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    if hasattr(gigaleak_gen, "_orchestrator") and gigaleak_gen._orchestrator:
        gigaleak_gen._orchestrator.generate = lambda prompt, **kw: hybrid_orch.generate(prompt, domain="gigaleak", **kw)
    curator.register_generator("gigaleak", gigaleak_gen)
    logger.info("✓ Gigaleak (GPT-5.2)")

    # Generate
    if not checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"oracle_farore_hybrid_{timestamp}"
        checkpoint = GenerationCheckpoint.create(output_name, target_count)

    logger.info(f"\nOutput: {output_name}")
    logger.info("Starting hybrid generation...\n")

    try:
        result = await curator.curate_dataset(
            domains=["oracle", "asm", "gigaleak"],
            target_count=target_count,
            quality_threshold=0.7,
            balance_domains=True,
            output_name=output_name,
            resume=checkpoint.completed_count > 0,
        )

        # Update checkpoint on success
        checkpoint.completed_count = result.stats.final_count
        checkpoint.domains_progress = result.stats.domain_counts
        checkpoint.provider_usage = hybrid_orch._request_count
        checkpoint.estimated_cost_usd = hybrid_orch._estimated_cost
        checkpoint.save()

    except KeyboardInterrupt:
        logger.warning("\nInterrupted! Saving checkpoint...")
        checkpoint.provider_usage = hybrid_orch._request_count
        checkpoint.estimated_cost_usd = hybrid_orch._estimated_cost
        checkpoint.save()
        logger.info(f"Resume with: --resume {output_name}")
        return 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        checkpoint.failed_items.append(str(e))
        checkpoint.provider_usage = hybrid_orch._request_count
        checkpoint.estimated_cost_usd = hybrid_orch._estimated_cost
        checkpoint.save()
        logger.info(f"Resume with: --resume {output_name}")
        raise

    # Stats
    stats = hybrid_orch.get_stats()

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total generated: {result.stats.total_generated}")
    logger.info(f"Final count: {result.stats.final_count}")
    logger.info(f"Acceptance rate: {result.stats.passed_quality / max(result.stats.total_generated, 1) * 100:.1f}%")
    logger.info(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")

    logger.info("\nProvider usage:")
    for provider, count in stats["request_counts"].items():
        health = stats["provider_health"][provider]
        logger.info(f"  {provider}: {count} requests ({health['success_rate']*100:.1f}% success)")

    logger.info("\nDomain breakdown:")
    for domain, count in result.stats.domain_counts.items():
        logger.info(f"  {domain}: {count}")

    logger.info("\nQuality scores:")
    for domain, score in result.stats.quality_scores.items():
        logger.info(f"  {domain}: {score:.3f}")

    logger.info("=" * 80)
    logger.info(f"\nDataset: {result.output_dir}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Oracle-Farore dataset with hybrid providers")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint name")
    parser.add_argument("--target", type=int, default=1000, help="Target sample count")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(resume_name=args.resume, target_count=args.target)))
