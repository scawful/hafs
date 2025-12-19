"""Test Knowledge Pipeline (Graph, Vector, Viz)."""

import asyncio
import os
import sys
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))

from hafs.agents.knowledge_graph import KnowledgeGraphAgent
from hafs.agents.vector_memory import ContextVectorAgent
from hafs.agents.visualizer import VisualizerAgent

async def test_pipeline():
    print("--- Testing Knowledge Pipeline ---")
    
    # 1. Test Knowledge Graph
    kg = KnowledgeGraphAgent()
    kg.orchestrator = MagicMock()
    
    # Mock LLM extraction
    mock_extract = {
        "nodes": [{"id": "TestNode", "type": "Concept"}],
        "edges": [{"source": "TestNode", "target": "OtherNode", "relation": "tests"}]
    }
    f_gen = asyncio.Future()
    f_gen.set_result(json.dumps(mock_extract))
    kg.orchestrator.generate_content.return_value = f_gen
    
    # Mock file system scan
    kg.verified_dir = Path("mock_verified")
    kg.verified_dir.mkdir(exist_ok=True)
    (kg.verified_dir / "test_doc.md").write_text("Test content for graph.")
    
    try:
        graph = await kg.build_graph()
        print(f"Graph Nodes: {len(graph['nodes'])}")
        if "TestNode" in graph['nodes']:
            print("✅ Knowledge Graph extraction working.")
        else:
            print("❌ Knowledge Graph failed.")
            
    finally:
        import shutil
        if kg.verified_dir.exists(): shutil.rmtree(kg.verified_dir)

    # 2. Test Vector Memory (Search)
    vec = ContextVectorAgent()
    # Mock model
    vec.model = MagicMock()
    vec.model.encode.return_value = [[0.1, 0.2]] # Mock embedding
    
    # Mock Index
    vec.index = {
        "embeddings": [[0.1, 0.2]],
        "metadata": [{"filename": "test.md", "content": "Test content", "path": "path/to/test.md"}]
    }
    
    res = await vec.search("query")
    if res and res[0]['filename'] == "test.md":
        print("✅ Vector Search working.")
    else:
        print("❌ Vector Search failed.")

    # 3. Test Visualizer
    viz = VisualizerAgent()
    viz.orchestrator = MagicMock()
    
    f_viz = asyncio.Future()
    f_viz.set_result("```mermaid\ngraph LR\nA-->B\n```")
    viz.orchestrator.generate_content.return_value = f_viz
    
    diag = await viz.create_diagram("Context")
    if diag['type'] == 'mermaid' and "A-->B" in diag['content']:
        print("✅ Visualizer working.")
    else:
        print("❌ Visualizer failed.")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
