#!/usr/bin/env python3
"""Test Pattern Analyzer agent."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.background.pattern_analyzer import PatternAnalyzerAgent


async def main():
    """Test pattern analyzer on Oracle-of-Secrets codebase."""
    print("=" * 80)
    print("Pattern Analyzer Test")
    print("=" * 80)
    print()

    # Initialize agent
    print("Setting up Pattern Analyzer...")
    agent = PatternAnalyzerAgent()
    await agent.setup()

    # Scan Oracle-of-Secrets for patterns
    oracle_path = Path.home() / "Code" / "Oracle-of-Secrets"
    if not oracle_path.exists():
        print(f"❌ Oracle path not found: {oracle_path}")
        return

    print(f"Scanning {oracle_path} for interesting patterns...")
    patterns = await agent.analyze_codebase(
        root=oracle_path,
        file_patterns=["**/*.asm"],
    )

    print(f"\n✓ Found {len(patterns)} code patterns")
    print()

    # Show top 5 patterns by pedagogical value
    top_patterns = sorted(
        patterns,
        key=lambda p: p.pedagogical_value * p.complexity_score,
        reverse=True,
    )[:5]

    print("Top 5 Patterns by Teaching Value:")
    print("-" * 80)
    for i, pattern in enumerate(top_patterns, 1):
        print(f"\n{i}. {pattern.pattern_type.upper()}")
        print(f"   File: {pattern.file_path}")
        print(f"   Line: {pattern.line_number}")
        print(f"   Complexity: {pattern.complexity_score:.2f}")
        print(f"   Pedagogical Value: {pattern.pedagogical_value:.2f}")
        print(f"   Code: {pattern.code_snippet[:100]}...")

    # Generate questions for top 3 patterns
    print("\n" + "=" * 80)
    print("Generating Expert Questions")
    print("=" * 80)

    questions = []
    for i, pattern in enumerate(top_patterns[:3], 1):
        print(f"\n[{i}/3] Generating question for {pattern.pattern_type}...")
        question = await agent.generate_question(pattern)

        if question:
            questions.append(question)
            print(f"✓ Generated: {question.question_id}")
            print(f"  Type: {question.question_type}")
            print(f"  Difficulty: {question.difficulty}")
            print(f"  Priority: {question.priority_score:.2f}")
            print(f"  Question: {question.question_text[:150]}...")
        else:
            print(f"✗ Failed to generate question")

    # Save questions
    if questions:
        print(f"\nSaving {len(questions)} questions to database...")
        await agent.save_questions(questions)
        print(f"✓ Saved to {agent.questions_db}")

        # Load back to verify
        loaded = await agent.load_questions()
        print(f"✓ Verified: {len(loaded)} questions in database")

    print("\n" + "=" * 80)
    print("Pattern Analyzer Test Complete")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Patterns detected: {len(patterns)}")
    print(f"  Questions generated: {len(questions)}")
    print(f"  Database: {agent.questions_db}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
