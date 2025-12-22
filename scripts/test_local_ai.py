#!/usr/bin/env python3
"""Test script for Local AI Orchestrator.

Quick test to verify:
1. OllamaClient connectivity
2. Tool execution
3. Priority queue ordering
4. Context integration
"""

import asyncio
import sys
from pathlib import Path

# Add hafs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.local_ai_orchestrator import (
    LocalAIOrchestrator,
    InferenceRequest,
    RequestPriority,
)
from services.ollama_client import OllamaClient
from services.tool_executor import AVAILABLE_TOOLS


async def test_ollama_connection():
    """Test 1: Check if Ollama is available."""
    print("\n" + "="*80)
    print("TEST 1: Ollama Connection")
    print("="*80)

    client = OllamaClient()
    available = await client.is_available_async()

    if available:
        print("✓ Ollama is running")

        models = client.list_models()
        print(f"✓ Found {len(models)} model(s):")
        for model in models:
            print(f"  - {model['name']}")

        return True
    else:
        print("✗ Ollama is NOT running")
        print("\nTo start Ollama:")
        print("  Mac: brew services start ollama")
        print("  Windows: Start Ollama Desktop app")
        print("  Linux: systemctl start ollama")
        return False


async def test_simple_inference():
    """Test 2: Simple text generation."""
    print("\n" + "="*80)
    print("TEST 2: Simple Inference")
    print("="*80)

    orch = LocalAIOrchestrator(default_model="qwen2.5:3b")
    await orch.start()

    request = InferenceRequest(
        id="test_simple",
        priority=RequestPriority.INTERACTIVE,
        prompt="Say 'Hello from Local AI!' in exactly 5 words.",
        model="qwen2.5:3b",
    )

    result = await orch.submit_request(request)

    if result.error:
        print(f"✗ Inference failed: {result.error}")
        await orch.stop()
        return False

    print(f"✓ Response: {result.response}")
    print(f"✓ Inference time: {result.inference_time_seconds:.2f}s")

    await orch.stop()
    return True


async def test_tool_calling():
    """Test 3: Tool calling functionality."""
    print("\n" + "="*80)
    print("TEST 3: Tool Calling")
    print("="*80)

    orch = LocalAIOrchestrator(default_model="qwen2.5:7b")
    await orch.start()

    # Create a test file
    test_file = Path.home() / ".context" / "test_ai_tools.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("This is a test file for AI tool calling.")

    request = InferenceRequest(
        id="test_tools",
        priority=RequestPriority.INTERACTIVE,
        prompt=f"Read the file {test_file} and tell me what it says.",
        model="qwen2.5:7b",
        tools=AVAILABLE_TOOLS,
    )

    result = await orch.submit_request(request)

    if result.error:
        print(f"✗ Tool calling failed: {result.error}")
        await orch.stop()
        return False

    print(f"✓ Response: {result.response[:200]}...")
    print(f"✓ Tool calls made: {len(result.tool_calls)}")

    for i, tool_call in enumerate(result.tool_calls, 1):
        print(f"  {i}. {tool_call.get('tool_name', 'unknown')}")

    # Cleanup
    test_file.unlink(missing_ok=True)

    await orch.stop()
    return True


async def test_priority_queue():
    """Test 4: Priority queue ordering."""
    print("\n" + "="*80)
    print("TEST 4: Priority Queue Ordering")
    print("="*80)

    orch = LocalAIOrchestrator(default_model="qwen2.5:3b")
    await orch.start()

    # Submit requests in reverse priority order
    requests = [
        InferenceRequest(
            id="low_priority",
            priority=RequestPriority.SCHEDULED,
            prompt="Low priority task",
            model="qwen2.5:3b",
        ),
        InferenceRequest(
            id="medium_priority",
            priority=RequestPriority.ANALYSIS,
            prompt="Medium priority task",
            model="qwen2.5:3b",
        ),
        InferenceRequest(
            id="high_priority",
            priority=RequestPriority.TRAINING,
            prompt="High priority task",
            model="qwen2.5:3b",
        ),
    ]

    # Submit all at once
    print("Submitting 3 requests (SCHEDULED, ANALYSIS, TRAINING)...")
    results = await asyncio.gather(*[orch.submit_request(r) for r in requests])

    # Check completion order (should be TRAINING first)
    print("✓ All requests completed")
    print(f"  1st completed: {results[2].request_id} (should be high_priority)")
    print(f"  2nd completed: {results[1].request_id}")
    print(f"  3rd completed: {results[0].request_id}")

    stats = orch.get_stats()
    print(f"\n✓ Orchestrator stats:")
    print(f"  Total: {stats['total_requests']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print(f"  Avg time: {stats['avg_inference_time_seconds']:.2f}s")

    await orch.stop()
    return True


async def test_context_integration():
    """Test 5: hafs context integration."""
    print("\n" + "="*80)
    print("TEST 5: Context Integration")
    print("="*80)

    orch = LocalAIOrchestrator(default_model="qwen2.5:7b")
    await orch.start()

    # Create test scratchpad
    scratchpad_dir = Path.home() / ".context" / "scratchpad"
    scratchpad_dir.mkdir(parents=True, exist_ok=True)

    state_file = scratchpad_dir / "state.json"
    state_file.write_text('{"test_key": "test_value", "active_tasks": ["testing"]}')

    request = InferenceRequest(
        id="test_context",
        priority=RequestPriority.INTERACTIVE,
        prompt="What is in my scratchpad state?",
        model="qwen2.5:7b",
        context_paths=["scratchpad"],
    )

    result = await orch.submit_request(request)

    if result.error:
        print(f"✗ Context integration failed: {result.error}")
        await orch.stop()
        return False

    print(f"✓ Response: {result.response[:200]}...")
    print(f"✓ Context tokens used: {result.context_tokens}")

    await orch.stop()
    return True


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LOCAL AI ORCHESTRATOR TEST SUITE")
    print("="*80)

    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Simple Inference", test_simple_inference),
        ("Tool Calling", test_tool_calling),
        ("Priority Queue", test_priority_queue),
        ("Context Integration", test_context_integration),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = await test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check Ollama installation and configuration.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
