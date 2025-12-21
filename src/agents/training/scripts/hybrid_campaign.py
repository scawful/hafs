#!/usr/bin/env python3
"""Hybrid GPU + API Training Campaign Launcher.

Runs training data generation using BOTH:
1. Your local GPU (medical-mechanica) - FREE
2. Gemini API - PAID

Automatically routes requests based on GPU load to maximize throughput
while minimizing API costs.

Usage:
    python -m agents.training.scripts.hybrid_campaign --target 34500
    python -m agents.training.scripts.hybrid_campaign --pilot
    python -m agents.training.scripts.hybrid_campaign --resume --target 34500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.training.hybrid_orchestrator import GPUMonitor, HybridLoadBalancer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class HybridOrchestrator:
    """Orchestrator that routes between GPU and API."""

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        load_balancer: HybridLoadBalancer,
    ):
        self.gpu_monitor = gpu_monitor
        self.load_balancer = load_balancer

        # Lazy-loaded backends
        self._gpu_backend = None
        self._api_backend = None

    async def setup(self):
        """Initialize backends."""
        try:
            from hafs.backends.ollama import OllamaBackend

            # GPU backend (Ollama on medical-mechanica)
            self._gpu_backend = OllamaBackend(
                base_url="http://100.104.53.21:11435",
                model="qwen3-14b",
            )
            logger.info("✓ GPU backend initialized (Ollama)")
        except Exception as e:
            logger.warning(f"GPU backend initialization failed: {e}")
            self._gpu_backend = None

        # API backend (Gemini) - use google-generativeai directly
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No GOOGLE_API_KEY found, API fallback disabled")
            self._api_backend = None
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._api_backend = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("✓ Gemini API backend available")
            except Exception as e:
                logger.warning(f"Gemini API initialization failed: {e}")
                self._api_backend = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using hybrid routing.

        Routes to GPU (free) or API (paid) based on current GPU load.
        """
        # Decide routing
        use_gpu, reason = await self.load_balancer.should_use_gpu()

        logger.debug(f"Routing: {reason}")

        # Try GPU first if selected
        if use_gpu and self._gpu_backend:
            try:
                response = await self._gpu_backend.generate(prompt, **kwargs)
                self.load_balancer.record_request(used_gpu=True, success=True)
                return response
            except Exception as e:
                logger.warning(f"GPU generation failed: {e}, falling back to API")
                self.load_balancer.record_request(used_gpu=True, success=False)
                use_gpu = False  # Fallback

        # Use API (fallback or primary choice)
        if self._api_backend:
            # Convert to sync for generativeai API
            response = self._api_backend.generate_content(prompt)
            self.load_balancer.record_request(used_gpu=False, success=True)
            return response.text
        else:
            raise RuntimeError("No available backend (GPU failed, API not configured)")


async def run_hybrid_campaign(
    target_count: int = 34500,
    output_name: Optional[str] = None,
    pilot: bool = False,
    resume: bool = False,
    enable_active_learning: bool = True,
    quality_threshold: Optional[float] = None,
):
    """Run training campaign with hybrid GPU + API routing.

    Args:
        target_count: Total samples to generate
        output_name: Custom output name
        pilot: Run pilot mode (max 1K samples)
        resume: Resume from checkpoint
        enable_active_learning: Use coverage-driven generation
        quality_threshold: Base quality threshold (None = domain-specific)

    Returns:
        CurationResult with generated samples and stats
    """
    start_time = datetime.now()

    # Pilot mode override
    if pilot:
        target_count = min(target_count, 1000)
        logger.info(f"PILOT MODE: Limiting to {target_count} samples")

    # Default output name
    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "pilot_hybrid" if pilot else "hybrid"
        output_name = f"{prefix}_{target_count}_{timestamp}"

    logger.info("=" * 80)
    logger.info("HYBRID GPU + API TRAINING CAMPAIGN")
    logger.info(f"Target: {target_count} samples")
    logger.info(f"Output: {output_name}")
    logger.info(f"Quality threshold: {quality_threshold}")
    logger.info(f"Active learning: {enable_active_learning}")
    logger.info(f"Resume: {resume}")
    logger.info("=" * 80)

    # Initialize hybrid system
    logger.info("\nInitializing hybrid GPU + API system...")
    gpu_monitor = GPUMonitor()
    load_balancer = HybridLoadBalancer(
        gpu_monitor, gpu_threshold_low=70.0, gpu_threshold_high=90.0
    )

    # Check GPU status
    gpu_status = await gpu_monitor.get_status()
    if gpu_status:
        logger.info(f"✓ GPU Online: {gpu_status.utilization_percent:.0f}% utilized")
        logger.info(
            f"  Memory: {gpu_status.memory_used_mb}MB / {gpu_status.memory_total_mb}MB"
        )
        logger.info(f"  Temperature: {gpu_status.temperature_c}°C")
    else:
        logger.warning("⚠️  GPU Unavailable - will use API only")

    # Initialize curator with hybrid orchestrator
    from agents.training.curator import DataCurator

    logger.info("\nInitializing DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Patch curator's orchestrator to use hybrid routing
    # (This is a bit hacky but allows us to plug in without rewriting everything)
    hybrid_orch = HybridOrchestrator(gpu_monitor, load_balancer)
    await hybrid_orch.setup()

    # Register all generators
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.oracle_generator import OracleDataGenerator
    from agents.training.generators.text_generator import TextDataGenerator

    logger.info("Registering generators with hybrid routing...")

    # ASM
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    # Patch orchestrator to use hybrid
    if hasattr(asm_gen, "_orchestrator") and asm_gen._orchestrator:
        asm_gen._orchestrator.generate = hybrid_orch.generate
    curator.register_generator("asm", asm_gen)
    logger.info("✓ ASM generator (hybrid: GPU or Gemini)")

    # Gigaleak
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    if hasattr(gigaleak_gen, "_orchestrator") and gigaleak_gen._orchestrator:
        gigaleak_gen._orchestrator.generate = hybrid_orch.generate
    curator.register_generator("gigaleak", gigaleak_gen)
    logger.info("✓ Gigaleak generator (hybrid: GPU or Gemini)")

    # Oracle
    oracle_gen = OracleDataGenerator()
    await oracle_gen.setup()
    if hasattr(oracle_gen, "_orchestrator") and oracle_gen._orchestrator:
        oracle_gen._orchestrator.generate = hybrid_orch.generate
    curator.register_generator("oracle", oracle_gen)
    logger.info("✓ Oracle generator (hybrid: GPU or Gemini)")

    # YAZE
    cpp_gen = CppDataGenerator()
    if cpp_gen.yaze_path.exists():
        await cpp_gen.setup()
        if hasattr(cpp_gen, "_orchestrator") and cpp_gen._orchestrator:
            cpp_gen._orchestrator.generate = hybrid_orch.generate
        curator.register_generator("yaze", cpp_gen)
        logger.info("✓ YAZE generator (hybrid: GPU or Gemini)")

    # Error
    error_gen = ErrorSampleGenerator(lookback_hours=168)
    await error_gen.setup()
    if hasattr(error_gen, "_orchestrator") and error_gen._orchestrator:
        error_gen._orchestrator.generate = hybrid_orch.generate
    curator.register_generator("errors", error_gen)
    logger.info("✓ Error generator (hybrid: GPU or Gemini)")

    # Text
    text_gen = TextDataGenerator()
    await text_gen.setup()
    if hasattr(text_gen, "_orchestrator") and text_gen._orchestrator:
        text_gen._orchestrator.generate = hybrid_orch.generate
    curator.register_generator("text", text_gen)
    logger.info("✓ Text generator (hybrid: GPU or Gemini)")

    logger.info(f"\nAll generators registered. Total domains: {len(curator.list_domains())}")

    # Run curation
    logger.info("\nStarting hybrid generation campaign...")
    logger.info("Routing: GPU <70% load → Gemini 70-90% → Gemini >90%")

    result = await curator.curate_dataset(
        domains=["asm", "gigaleak", "oracle", "yaze", "errors", "text"],
        target_count=target_count,
        quality_threshold=quality_threshold,
        balance_domains=True,
        output_name=output_name,
        enable_active_learning=enable_active_learning,
        resume=resume,
    )

    # Show routing stats
    logger.info("\n" + "=" * 80)
    logger.info("CAMPAIGN COMPLETE")
    logger.info(f"Total generated: {result.stats.total_generated}")
    logger.info(f"Final count: {result.stats.final_count}")

    stats = load_balancer.get_stats()
    logger.info("\nRouting Statistics:")
    logger.info(f"  GPU requests: {stats['gpu_requests']} ({stats['gpu_percentage']:.1f}%)")
    logger.info(f"  API requests: {stats['api_requests']} ({100-stats['gpu_percentage']:.1f}%)")
    logger.info(f"  GPU failures: {stats['gpu_failures']}")

    # Cost estimate
    api_cost_per_sample = 0.001  # Rough estimate
    estimated_cost = stats["api_requests"] * api_cost_per_sample
    estimated_savings = stats["gpu_requests"] * api_cost_per_sample
    logger.info(f"\nCost Estimates:")
    logger.info(f"  API cost: ${estimated_cost:.2f}")
    logger.info(f"  GPU savings: ${estimated_savings:.2f}")

    duration = datetime.now() - start_time
    logger.info(f"\nDuration: {duration}")
    logger.info("=" * 80)

    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid GPU + API training campaign"
    )
    parser.add_argument(
        "--target", type=int, default=34500, help="Target number of samples"
    )
    parser.add_argument("--output-name", type=str, help="Custom output directory name")
    parser.add_argument(
        "--pilot", action="store_true", help="Run pilot mode (max 1K samples)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoints if available"
    )
    parser.add_argument(
        "--no-active-learning",
        action="store_true",
        help="Disable coverage-driven active learning",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Base quality threshold (default: None, uses domain-specific)",
    )

    args = parser.parse_args()

    result = await run_hybrid_campaign(
        target_count=args.target,
        output_name=args.output_name,
        pilot=args.pilot,
        resume=args.resume,
        enable_active_learning=not args.no_active_learning,
        quality_threshold=args.quality_threshold,
    )

    logger.info("✓ Campaign complete!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
