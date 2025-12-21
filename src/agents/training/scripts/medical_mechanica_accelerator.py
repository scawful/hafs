#!/usr/bin/env python3
"""Medical-Mechanica GPU Acceleration for Training Data Generation.

Offloads compute-intensive operations to the medical-mechanica Windows GPU node:
1. Teacher model inference (Ollama on GPU)
2. Embedding generation (local embedding models)
3. Parallel generation streams
4. Quality validation

This replaces expensive Gemini API calls with free local inference on the 5060TI 16GB.

Usage:
    python -m agents.training.scripts.medical_mechanica_accelerator --pilot
    python -m agents.training.scripts.medical_mechanica_accelerator --target 34500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from hafs.core.orchestrator_v2 import Provider, TaskTier, UnifiedOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def configure_medical_mechanica_routing(orchestrator: UnifiedOrchestrator):
    """Configure orchestrator to prefer medical-mechanica for all local-capable tasks.

    Routes:
    - CODING tier → qwen2.5-coder:14b on medical-mechanica (Ollama)
    - REASONING tier → deepseek-r1:8b on medical-mechanica (Ollama)
    - FAST tier → gemma3:4b on medical-mechanica (Ollama)
    - Embeddings → nomic-embed-text on medical-mechanica (Ollama)

    Fallback to Gemini API only if medical-mechanica is unavailable.
    """
    logger.info("Configuring medical-mechanica GPU acceleration...")

    # Check medical-mechanica availability
    node_manager = orchestrator._node_manager
    if not node_manager:
        logger.warning("NodeManager not initialized")
        return False

    mm_node = node_manager.get_node("medical-mechanica")

    if not mm_node:
        logger.warning("medical-mechanica node not found in nodes.toml")
        logger.warning("Falling back to Gemini API")
        return False

    # Check if reachable
    is_healthy = await node_manager.health_check(mm_node)
    if not is_healthy:
        logger.warning(f"medical-mechanica not reachable at {mm_node.host}:{mm_node.port}")
        logger.warning("Ensure Ollama is running and Tailscale is connected")
        logger.warning("Falling back to Gemini API")
        return False

    logger.info(f"✓ medical-mechanica online at {mm_node.host}:{mm_node.port}")
    logger.info(f"✓ Available models: {mm_node.models}")

    # Override tier routing to prefer Ollama on medical-mechanica
    orchestrator._prefer_local = True
    orchestrator._preferred_node = "medical-mechanica"

    logger.info("✓ GPU acceleration enabled")
    logger.info("  CODING → qwen2.5-coder:14b (medical-mechanica)")
    logger.info("  REASONING → deepseek-r1:8b (medical-mechanica)")
    logger.info("  FAST → gemma3:4b (medical-mechanica)")
    logger.info("  Embeddings → nomic-embed-text (medical-mechanica)")

    return True


async def run_parallel_generation(
    target_count: int,
    output_name: str,
    use_gpu: bool = True,
):
    """Run parallel generation campaign with GPU acceleration.

    Launches multiple generator streams in parallel, each using
    medical-mechanica GPU for local inference.

    Args:
        target_count: Total samples to generate
        output_name: Output directory name
        use_gpu: Use medical-mechanica GPU (default: True)
    """
    from agents.training.curator import DataCurator

    logger.info("=" * 80)
    logger.info("MEDICAL-MECHANICA GPU ACCELERATION MODE")
    logger.info(f"Target: {target_count} samples")
    logger.info(f"GPU Acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
    logger.info("=" * 80)

    # Initialize curator
    curator = DataCurator()
    await curator.setup()

    # Configure GPU acceleration
    if use_gpu:
        gpu_enabled = await configure_medical_mechanica_routing(
            curator._quality_pipeline.orchestrator
        )
        if not gpu_enabled:
            logger.warning("GPU acceleration failed, using Gemini API fallback")
    else:
        logger.info("GPU acceleration disabled, using Gemini API")

    # Register all generators
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.oracle_generator import OracleDataGenerator
    from agents.training.generators.text_generator import TextDataGenerator

    logger.info("Registering generators...")

    # ASM Generator
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    if use_gpu and asm_gen._orchestrator:
        await configure_medical_mechanica_routing(asm_gen._orchestrator)
    curator.register_generator("asm", asm_gen)
    logger.info("✓ ASM generator (qwen2.5-coder:14b)")

    # Gigaleak Generator
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    if use_gpu and gigaleak_gen._orchestrator:
        await configure_medical_mechanica_routing(gigaleak_gen._orchestrator)
    curator.register_generator("gigaleak", gigaleak_gen)
    logger.info("✓ Gigaleak generator (qwen2.5-coder:14b)")

    # Oracle Generator
    oracle_gen = OracleDataGenerator()
    await oracle_gen.setup()
    if use_gpu and oracle_gen._orchestrator:
        await configure_medical_mechanica_routing(oracle_gen._orchestrator)
    curator.register_generator("oracle", oracle_gen)
    logger.info("✓ Oracle generator (qwen2.5-coder:14b)")

    # YAZE Generator
    cpp_gen = CppDataGenerator()
    if cpp_gen.yaze_path.exists():
        await cpp_gen.setup()
        if use_gpu and cpp_gen._orchestrator:
            await configure_medical_mechanica_routing(cpp_gen._orchestrator)
        curator.register_generator("yaze", cpp_gen)
        logger.info("✓ YAZE generator (deepseek-coder-v2-lite)")

    # Error Generator
    error_gen = ErrorSampleGenerator(lookback_hours=168)
    await error_gen.setup()
    if use_gpu and error_gen._orchestrator:
        await configure_medical_mechanica_routing(error_gen._orchestrator)
    curator.register_generator("errors", error_gen)
    logger.info("✓ Error generator (deepseek-r1:8b)")

    # Text Generator
    text_gen = TextDataGenerator()
    await text_gen.setup()
    if use_gpu and text_gen._orchestrator:
        await configure_medical_mechanica_routing(text_gen._orchestrator)
    curator.register_generator("text", text_gen)
    logger.info("✓ Text generator (gemma3:4b)")

    logger.info(f"All generators configured for {'GPU' if use_gpu else 'API'} mode")

    # Run curation
    result = await curator.curate_dataset(
        domains=["asm", "gigaleak", "oracle", "yaze", "errors", "text"],
        target_count=target_count,
        quality_threshold=0.7,
        balance_domains=True,
        output_name=output_name,
    )

    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info(f"Total generated: {result.stats.total_generated}")
    logger.info(f"Final count: {result.stats.final_count}")
    logger.info("=" * 80)

    return result


async def test_medical_mechanica_connection():
    """Test connection to medical-mechanica and available models."""
    logger.info("Testing medical-mechanica connection...")

    orchestrator = UnifiedOrchestrator()

    # Get medical-mechanica node
    if not orchestrator._node_manager:
        logger.error("NodeManager not initialized")
        return False

    mm_node = orchestrator._node_manager.get_node("medical-mechanica")
    if not mm_node:
        logger.error("medical-mechanica node not configured")
        logger.error("Add to hafs.toml:")
        logger.error("""
[[training.nodes]]
name = "medical-mechanica"
host = "100.100.100.20"
port = 11434
gpu = "5060TI"
memory_gb = 16
models = ["qwen2.5-coder:14b", "deepseek-r1:8b", "gemma3:4b"]
""")
        return False

    logger.info(f"Node: {mm_node.name}")
    logger.info(f"Host: {mm_node.host}:{mm_node.port}")
    logger.info(f"GPU: {mm_node.gpu_memory_mb / 1024:.1f} GB")

    # Test health
    is_healthy = await orchestrator._node_manager.health_check(mm_node)
    if is_healthy:
        logger.info("✓ Connection successful")
        logger.info(f"✓ Available models: {mm_node.models}")
        return True
    else:
        logger.error("✗ Connection failed")
        logger.error("Troubleshooting:")
        logger.error("1. Ensure Tailscale is connected")
        logger.error("2. Ensure Ollama is running on Windows: ollama serve")
        logger.error("3. Test connection: curl http://100.100.100.20:11434/api/tags")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="medical-mechanica GPU acceleration for training data generation"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target number of samples (default: 1000)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom output directory name",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode (max 1K samples)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test medical-mechanica connection only",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use Gemini API)",
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        success = await test_medical_mechanica_connection()
        return 0 if success else 1

    # Pilot mode
    if args.pilot:
        args.target = min(args.target, 1000)

    # Default output name
    if not args.output_name:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "pilot_gpu" if args.pilot else "full_gpu"
        args.output_name = f"{prefix}_{args.target}_{timestamp}"

    # Run generation
    result = await run_parallel_generation(
        target_count=args.target,
        output_name=args.output_name,
        use_gpu=not args.no_gpu,
    )

    logger.info("Campaign complete!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
