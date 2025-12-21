#!/usr/bin/env python3
"""Test script for Mixture of Experts system."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def test_classifier():
    """Test the task classifier."""
    from hafs.agents.moe import TaskClassifier

    logger.info("=" * 80)
    logger.info("TEST 1: Task Classifier")
    logger.info("=" * 80)

    classifier = TaskClassifier()
    await classifier.initialize()

    test_tasks = [
        "Write a routine to add a new item to ALTTP",
        "Use YAZE to replace Link's sprite graphics",
        "Debug why my ROM hack crashes after talking to an NPC",
        "Create a custom sprite that loads from YAZE and uses optimized assembly",
    ]

    for task in test_tasks:
        logger.info(f"\nTask: {task}")
        classification = await classifier.classify(task)
        logger.info(f"Experts: {classification.expert_names}")
        logger.info(f"Confidences: {[f'{c:.2f}' for c in classification.confidences]}")
        logger.info(f"Multi-expert: {classification.is_multi_expert}")
        logger.info(f"Reasoning: {classification.reasoning}")

    logger.info("\n✓ Classifier tests completed")


async def test_single_expert():
    """Test single expert execution."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Single Expert Execution")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    # Test ASM Expert
    logger.info("\n[ASM Expert Test]")
    result = await orchestrator.execute(
        "Write a simple 65816 assembly routine that checks if A register is zero"
    )

    logger.info(f"Experts used: {result.experts_used}")
    logger.info(f"Synthesis used: {result.synthesis_used}")
    logger.info(f"\nResponse:\n{result.content[:500]}...")

    logger.info("\n✓ Single expert test completed")


async def test_multi_expert():
    """Test multi-expert execution with synthesis."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Multi-Expert Execution (ASM + YAZE)")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    # Complex task requiring both ASM and YAZE experts
    task = """
    Create a new custom item in ALTTP that:
    1. Has a unique item ID
    2. Uses custom graphics loaded from YAZE
    3. Has an assembly routine to handle item usage
    """

    logger.info(f"\nTask: {task}")

    result = await orchestrator.execute(task)

    logger.info(f"Experts used: {result.experts_used}")
    logger.info(f"Synthesis used: {result.synthesis_used}")
    logger.info(f"Total tokens: {result.metadata['total_tokens']}")
    logger.info(f"Total latency: {result.metadata['total_latency_ms']}ms")

    logger.info(f"\nSynthesized response:\n{result.content[:800]}...")

    logger.info("\n✓ Multi-expert test completed")


async def test_debug_expert():
    """Test debug expert."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Debug Expert")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    # Debug task
    task = """
    My ROM hack crashes when Link picks up a custom item.
    The screen freezes and I have to reset the emulator.
    I recently added a new item at slot $20 with custom graphics.
    """

    logger.info(f"\nTask: {task}")

    result = await orchestrator.execute(task)

    logger.info(f"Experts used: {result.experts_used}")
    logger.info(f"\nDebug response:\n{result.content[:800]}...")

    logger.info("\n✓ Debug expert test completed")


async def test_expert_routing_explanation():
    """Test routing explanation without execution."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Expert Routing Explanation")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    task = "Add a new dungeon room with custom tiles and enemy spawns"

    explanation = await orchestrator.explain_routing(task)
    logger.info(f"\n{explanation}")

    logger.info("\n✓ Routing explanation test completed")


async def test_forced_experts():
    """Test forcing specific experts."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Forced Expert Selection")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    task = "Explain how to handle sprite graphics"

    # Force both ASM and YAZE experts
    logger.info(f"\nTask: {task}")
    logger.info("Forcing experts: ['asm', 'yaze']")

    result = await orchestrator.execute(
        task,
        force_experts=["asm", "yaze"]
    )

    logger.info(f"Experts used: {result.experts_used}")
    logger.info(f"Synthesis used: {result.synthesis_used}")

    logger.info("\n✓ Forced expert test completed")


async def test_configurable_parameters():
    """Test configurable tokens and temperature."""
    from hafs.agents.moe import MoEOrchestrator, TaskClassifier, Synthesizer

    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: Configurable Parameters")
    logger.info("=" * 80)

    # Create orchestrator with custom parameters
    classifier = TaskClassifier(
        max_tokens=150,
        temperature=0.2,  # Very low for deterministic classification
    )
    synthesizer = Synthesizer(
        max_tokens=2048,
        temperature=0.8,  # Higher for creative synthesis
    )

    orchestrator = MoEOrchestrator()
    orchestrator.classifier = classifier
    orchestrator.synthesizer = synthesizer

    await orchestrator.initialize()

    logger.info("✓ Custom parameters configured:")
    logger.info(f"  Classifier: max_tokens={classifier.max_tokens}, temp={classifier.temperature}")
    logger.info(f"  Synthesizer: max_tokens={synthesizer.max_tokens}, temp={synthesizer.temperature}")

    # Also test expert-level configuration
    from hafs.agents.moe.experts import AsmExpert

    custom_asm_expert = AsmExpert(
        model_name="custom-asm-model",
        lora_adapter_path=Path("~/.context/models/custom/adapters"),
    )
    await custom_asm_expert.initialize()

    logger.info(f"  ASM Expert: {custom_asm_expert.config.model_name}")
    logger.info(f"    max_tokens={custom_asm_expert.config.max_tokens}")
    logger.info(f"    temperature={custom_asm_expert.config.temperature}")

    logger.info("\n✓ Configurable parameters test completed")


async def test_expert_info():
    """Test listing expert information."""
    from hafs.agents.moe import MoEOrchestrator

    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: Expert Information")
    logger.info("=" * 80)

    orchestrator = MoEOrchestrator()
    await orchestrator.initialize()

    # List all experts
    experts = await orchestrator.list_experts()
    logger.info("\nAvailable Experts:")
    for name, specialization in experts.items():
        logger.info(f"  - {name}: {specialization}")

    # Get detailed info for ASM expert
    logger.info("\nASM Expert Details:")
    info = await orchestrator.get_expert_info("asm")
    if info:
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

    logger.info("\n✓ Expert info test completed")


async def main():
    """Run all tests."""
    logger.info("MIXTURE OF EXPERTS SYSTEM TEST SUITE")
    logger.info("=" * 80)

    try:
        # Run tests sequentially
        await test_classifier()
        await test_single_expert()
        await test_multi_expert()
        await test_debug_expert()
        await test_expert_routing_explanation()
        await test_forced_experts()
        await test_configurable_parameters()
        await test_expert_info()

        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        logger.info("\nMoE System Summary:")
        logger.info("  - 3 Expert Agents (ASM, YAZE, Debug)")
        logger.info("  - Task Classifier (keyword + LLM)")
        logger.info("  - Multi-expert Synthesizer")
        logger.info("  - Configurable parameters (tokens, temperature)")
        logger.info("  - Parallel expert execution")
        logger.info("  - Ready for fine-tuned model integration")

        return 0

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
