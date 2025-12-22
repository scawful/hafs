#!/usr/bin/env python3
"""Test new models for training data generation.

Compare DeepSeek-R1, Qwen3, Gemma3 against current baseline.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.local_ai_orchestrator import (
    LocalAIOrchestrator,
    InferenceRequest,
    RequestPriority,
)


async def test_model(model: str, test_prompt: str) -> dict:
    """Test a single model.

    Args:
        model: Model name
        test_prompt: Test prompt

    Returns:
        Results dict with response, time, quality estimate
    """
    print(f"\n{'='*80}")
    print(f"Testing: {model}")
    print(f"{'='*80}")

    orch = LocalAIOrchestrator(
        ollama_url="http://100.104.53.21:11434",  # Windows machine
        default_model=model,
    )

    await orch.start()

    request = InferenceRequest(
        id=f"test_{model}_{datetime.now().timestamp()}",
        priority=RequestPriority.INTERACTIVE,
        prompt=test_prompt,
        model=model,
        max_tokens=2048,
        temperature=0.7,
    )

    start = datetime.now()
    result = await orch.submit_request(request)
    elapsed = (datetime.now() - start).total_seconds()

    await orch.stop()

    if result.error:
        print(f"❌ Error: {result.error}")
        return {"model": model, "error": result.error}

    print(f"✓ Response ({elapsed:.1f}s):")
    print(result.response[:500])
    if len(result.response) > 500:
        print(f"... ({len(result.response)} total chars)")

    return {
        "model": model,
        "response": result.response,
        "time_seconds": elapsed,
        "length_chars": len(result.response),
        "length_words": len(result.response.split()),
    }


async def main():
    """Run model comparison tests."""

    # Test prompt - ASM code explanation (typical training task)
    test_prompt = """Analyze this 65816 assembly routine and explain what it does:

```asm
Module07_02_InitSpriteData:
    PHB
    PHK
    PLB
    REP #$30
    LDX.w #$0000
.loop
    LDA.w Module07_SpriteData,X
    STA.w $7E2000,X
    INX
    INX
    CPX.w #$0200
    BNE .loop
    PLB
    RTL
```

Provide a clear, technical explanation of:
1. What this routine accomplishes
2. The purpose of each instruction group
3. Why certain operations are used (PHB/PLB, REP #$30, etc.)

Keep your explanation concise but technically accurate."""

    # Models to test
    models_to_test = [
        # Reasoning models
        ("deepseek-r1:8b", "DeepSeek-R1 8B (Reasoning)"),
        ("deepseek-r1:14b", "DeepSeek-R1 14B (Reasoning)"),

        # Latest generation
        ("qwen3:14b", "Qwen 3 14B (Latest)"),
        ("gemma3:12b", "Gemma 3 12B (Latest)"),

        # Current baseline
        ("qwen2.5-coder:14b", "Qwen 2.5-Coder 14B (Current)"),

        # Code specialists
        ("deepseek-coder:33b", "DeepSeek-Coder 33B"),
    ]

    results = []

    for model, description in models_to_test:
        print(f"\n\n{'#'*80}")
        print(f"# {description}")
        print(f"{'#'*80}")

        try:
            result = await test_model(model, test_prompt)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"model": model, "error": str(e)})

        # Brief pause between tests
        await asyncio.sleep(2)

    # Summary
    print("\n\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    successful = [r for r in results if "error" not in r]

    if not successful:
        print("❌ No models succeeded")
        return 1

    # Sort by speed
    successful.sort(key=lambda r: r["time_seconds"])

    print("\n⚡ Speed Ranking:")
    for i, r in enumerate(successful, 1):
        words_per_sec = r["length_words"] / r["time_seconds"]
        print(f"{i}. {r['model']:30s} {r['time_seconds']:6.1f}s  ({words_per_sec:.1f} words/sec)")

    # Sort by detail (length)
    successful.sort(key=lambda r: r["length_words"], reverse=True)

    print("\n📝 Detail Ranking:")
    for i, r in enumerate(successful, 1):
        print(f"{i}. {r['model']:30s} {r['length_words']:5d} words")

    print("\n\n💡 Recommendations:")

    # Fastest
    fastest = min(successful, key=lambda r: r["time_seconds"])
    print(f"- **Fastest:** {fastest['model']} ({fastest['time_seconds']:.1f}s)")

    # Most detailed
    most_detailed = max(successful, key=lambda r: r["length_words"])
    print(f"- **Most Detailed:** {most_detailed['model']} ({most_detailed['length_words']} words)")

    # Best balance (words/sec)
    best_balance = max(successful, key=lambda r: r["length_words"] / r["time_seconds"])
    wps = best_balance["length_words"] / best_balance["time_seconds"]
    print(f"- **Best Balance:** {best_balance['model']} ({wps:.1f} words/sec)")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
