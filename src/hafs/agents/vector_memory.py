"""Vector Memory Agent (Public Port).

Provides semantic search over verified knowledge using local embeddings.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from hafs.agents.base import BaseAgent
from hafs.core.orchestrator import ModelOrchestrator

class ContextVectorAgent(BaseAgent):
    """Indexes verified context and provides semantic search."""

    def __init__(self):
        super().__init__("ContextVectorAgent", "Semantic search for HAFS context.")
        self.model_name = "all-MiniLM-L6-v2" # Fast, efficient local model
        self.model = None
        self.index_path = self.context_root / "memory" / "vector_index.pkl"
        self.verified_dir = self.knowledge_dir / "verified"
        self.index = {"embeddings": [], "metadata": []}
        self.orchestrator = ModelOrchestrator()

    async def setup(self):
        """Load model and index."""
        print(f"[{self.name}] Loading embedding model {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"[{self.name}] Failed to load model: {e}")
            return

        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    self.index = pickle.load(f)
            except Exception as e:
                print(f"[{self.name}] Failed to load index: {e}")

    async def run_indexing_task(self):
        """Re-index all verified files."""
        if not self.model: return "Model not loaded."
        
        new_metadata = []
        new_docs = []
        
        if self.verified_dir.exists():
            for f in self.verified_dir.glob("*.md"):
                try:
                    text = f.read_text()
                    chunks = self._chunk_markdown(text)
                    for i, chunk in enumerate(chunks):
                        new_docs.append(chunk['text'])
                        new_metadata.append({
                            "path": str(f),
                            "filename": f.name,
                            "chunk_id": i,
                            "header": chunk['header'],
                            "content": chunk['text']
                        })
                except Exception as e:
                    print(f"Error reading {f}: {e}")
        
        if new_docs:
            embeddings = self.model.encode(new_docs)
            self.index = {"embeddings": embeddings, "metadata": new_metadata}
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
            return f"Indexed {len(new_docs)} chunks."
        
        return "No documents found."

    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search."""
        if not self.model or len(self.index["embeddings"]) == 0:
            return []
            
        query_embedding = self.model.encode([query])[0]
        embeddings = self.index["embeddings"]
        scores = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            meta = self.index["metadata"][idx]
            results.append({
                "score": float(scores[idx]),
                "filename": meta.get("filename", "unknown"),
                "content": meta.get("content", ""),
                "path": meta.get("path", "")
            })
        return results

    def _chunk_markdown(self, text: str) -> List[Dict]:
        chunks = []
        lines = text.splitlines()
        current_header = "Root"
        current_text = []
        for line in lines:
            if line.startswith("#"):
                if current_text:
                    chunks.append({"header": current_header, "text": "\n".join(current_text)})
                current_header = line.strip().lstrip("#").strip()
                current_text = [line]
            else:
                current_text.append(line)
        if current_text:
            chunks.append({"header": current_header, "text": "\n".join(current_text)})
        return chunks
