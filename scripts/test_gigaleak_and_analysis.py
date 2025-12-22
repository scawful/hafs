#!/usr/bin/env python3
"""Test script for GigaleakKB and Embedding Analysis.

Usage:
    # Test Gigaleak KB
    python scripts/test_gigaleak_and_analysis.py --test gigaleak

    # Build Gigaleak KB
    python scripts/test_gigaleak_and_analysis.py --build-gigaleak

    # Test embedding clustering
    python scripts/test_gigaleak_and_analysis.py --test clustering

    # Full analysis with Gemini 3
    python scripts/test_gigaleak_and_analysis.py --analyze
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def test_gigaleak_kb():
    """Test the GigaleakKB."""
    from agents.knowledge.gigaleak import GigaleakKB

    print_header("Testing GigaleakKB")

    kb = GigaleakKB()
    await kb.setup()

    stats = kb.get_statistics()
    print("Current Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if stats["total_symbols"] == 0:
        print("\nKB is empty. Run with --build-gigaleak to build.")
    else:
        print(f"\nLoaded {stats['total_symbols']} symbols")

        # Show sample modules
        print("\nSample Modules:")
        for name, mod in list(kb._modules.items())[:5]:
            print(f"  {name}: {len(mod.symbols)} globals, {len(mod.externals)} externals")

        # Show sample symbols
        print("\nSample Symbols:")
        for name, sym in list(kb._symbols.items())[:10]:
            jp = sym.japanese_comment[:30] if sym.japanese_comment else ""
            en = sym.english_translation[:30] if sym.english_translation else ""
            print(f"  {name} ({sym.symbol_type}): {jp or en or '(no comment)'}")


async def build_gigaleak_kb(translate: bool = True, embeddings: bool = True):
    """Build the GigaleakKB."""
    from agents.knowledge.gigaleak import GigaleakKB

    print_header("Building GigaleakKB")

    kb = GigaleakKB()
    await kb.setup()

    print(f"Source: {kb.source_path}")
    print(f"Building with:")
    print(f"  - Translate Japanese: {translate}")
    print(f"  - Generate embeddings: {embeddings}")
    print()

    stats = await kb.build(
        generate_embeddings=embeddings,
        translate_japanese=translate,
        batch_size=20,
    )

    print("\nBuild Complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def test_clustering():
    """Test embedding clustering."""
    from agents.analysis.embedding_analyzer import EmbeddingAnalyzer

    print_header("Testing Embedding Clustering")

    analyzer = EmbeddingAnalyzer()
    await analyzer.setup()

    # Load ALTTP embeddings
    print("Loading ALTTP embeddings...")
    count = await analyzer.load_from_kb("alttp")
    print(f"Loaded {count} embeddings")

    if count == 0:
        print("No embeddings found. Run embedding generation first.")
        return

    # Cluster
    print("\nClustering with K-Means (10 clusters)...")
    clusters = await analyzer.cluster(n_clusters=10, method="kmeans")

    print(f"\nCreated {len(clusters)} clusters:")
    for cluster in clusters:
        print(f"  Cluster {cluster.id}: {cluster.size} members, coherence: {cluster.coherence:.3f}")
        print(f"    Representatives: {', '.join(cluster.representative_members[:3])}")

    # Find similar items
    if analyzer._embeddings:
        sample_id = list(analyzer._embeddings.keys())[0]
        print(f"\nFinding items similar to '{sample_id}'...")
        result = analyzer.find_similar(sample_id, limit=5)
        print(f"Similar items:")
        for item_id, score in result.similar_items:
            print(f"  {item_id}: {score:.3f}")

    # Detect outliers
    print("\nDetecting outliers...")
    outliers = analyzer.detect_outliers(threshold=2.0)
    print(f"Found {len(outliers)} outliers")
    for outlier in outliers[:5]:
        print(f"  {outlier.outlier_id}: {outlier.reason}")


async def full_analysis():
    """Full analysis with Gemini 3 interpretation."""
    from agents.analysis.embedding_analyzer import EmbeddingAnalyzer

    print_header("Full Embedding Analysis with Gemini 3")

    analyzer = EmbeddingAnalyzer()
    await analyzer.setup()

    # Load all available embeddings
    print("Loading embeddings from all KBs...")
    total = 0
    for kb_name in ["alttp", "oracle-of-secrets", "gigaleak"]:
        count = await analyzer.load_from_kb(kb_name)
        if count > 0:
            print(f"  {kb_name}: {count}")
            total += count

    if total == 0:
        print("No embeddings found.")
        return

    print(f"\nTotal: {total} embeddings")

    # Cluster
    print("\nClustering...")
    n_clusters = min(15, total // 20 + 1)
    clusters = await analyzer.cluster(n_clusters=n_clusters)
    print(f"Created {len(clusters)} clusters")

    # Interpret with Gemini 3
    print("\nInterpreting clusters with Gemini 3...")
    interpretations = await analyzer.interpret_clusters()

    print("\nCluster Interpretations:")
    for cluster in analyzer._clusters:
        print(f"\n  Cluster {cluster.id} ({cluster.size} members):")
        print(f"    Label: {cluster.label}")
        print(f"    Description: {cluster.description}")
        print(f"    Top members: {', '.join(cluster.representative_members[:3])}")

    # Reduce dimensions and export
    print("\nReducing dimensions for visualization...")
    analyzer.reduce_dimensions(method="pca", n_components=2)

    output_path = Path.home() / ".context" / "analysis" / "alttp_visualization.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz_data = analyzer.export_for_visualization(output_path)

    print(f"\nExported visualization data to: {output_path}")
    print(f"  Points: {len(viz_data['points'])}")
    print(f"  Clusters: {len(viz_data['clusters'])}")


async def test_gemini3():
    """Test Gemini 3 models."""
    from core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

    print_header("Testing Gemini 3 Models")

    orchestrator = UnifiedOrchestrator(log_thoughts=True)
    await orchestrator.initialize()

    print("Available Gemini models:")
    for model, desc in orchestrator.GEMINI_MODELS.items():
        print(f"  {model}: {desc}")

    # Test generation
    print("\nTesting Gemini 3 Flash (via FAST tier)...")
    try:
        result = await orchestrator.generate(
            prompt="Explain in one sentence what A Link to the Past is.",
            tier=TaskTier.FAST,
            provider=Provider.GEMINI,
        )
        print(f"Response: {result.content}")
        print(f"Model used: {result.model}")
        print(f"Provider: {result.provider}")
        print(f"Tokens used: {result.tokens_used}")
        if result.thought_content:
            print(f"Thought trace ({len(result.thought_content)} chars):")
            print(f"  {result.thought_content[:300]}...")
        else:
            print("No thought trace in response")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure GEMINI_API_KEY is set and model is available")

    print("\nTesting Gemini 3 Pro (via REASONING tier)...")
    try:
        result = await orchestrator.generate(
            prompt="What makes the Master Sword special in Zelda lore?",
            tier=TaskTier.REASONING,
            provider=Provider.GEMINI,
        )
        print(f"Response: {result.content[:200]}...")
        print(f"Model used: {result.model}")
        print(f"Tokens used: {result.tokens_used}")
        if result.thought_content:
            print(f"Thought trace ({len(result.thought_content)} chars):")
            print(f"  {result.thought_content[:300]}...")
        else:
            print("No thought trace in response")
    except Exception as e:
        print(f"Error: {e}")

    # Check history for logged thought traces
    print("\nChecking history for thought traces...")
    try:
        from core.history.logger import HistoryLogger
        from core.history.models import OperationType
        history_dir = Path.home() / ".context" / "history"
        if history_dir.exists():
            logger = HistoryLogger(history_dir)
            from core.history.models import HistoryQuery
            recent = logger.query(HistoryQuery(
                operation_types=[OperationType.THOUGHT_TRACE],
                limit=5
            ))
            print(f"Found {len(recent)} thought trace entries in history")
            for entry in recent[:2]:
                print(f"  - {entry.operation.name}: {entry.timestamp[:19]}")
                if entry.operation.input:
                    thought = entry.operation.input.get("thought_content", "")
                    if thought:
                        print(f"    Thought: {thought[:100]}...")
    except Exception as e:
        print(f"Error checking history: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Test GigaleakKB and Embedding Analysis")

    parser.add_argument("--test", choices=["gigaleak", "clustering", "gemini3"],
                       help="Run specific test")
    parser.add_argument("--build-gigaleak", action="store_true",
                       help="Build GigaleakKB from source")
    parser.add_argument("--analyze", action="store_true",
                       help="Run full analysis with Gemini 3")
    parser.add_argument("--no-translate", action="store_true",
                       help="Skip Japanese translation")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Skip embedding generation")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"#  ALTTP Knowledge Base & Embedding Analysis")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if args.test == "gigaleak":
        await test_gigaleak_kb()
    elif args.test == "clustering":
        await test_clustering()
    elif args.test == "gemini3":
        await test_gemini3()
    elif args.build_gigaleak:
        await build_gigaleak_kb(
            translate=not args.no_translate,
            embeddings=not args.no_embeddings,
        )
    elif args.analyze:
        await full_analysis()
    else:
        # Default: run all tests
        await test_gemini3()
        await test_gigaleak_kb()
        await test_clustering()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
