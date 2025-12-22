"""Content Indexer Agent.

Crawls websites and generates embeddings for semantic search.
Creates knowledge base with page content chunked and embedded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from agents.background.base import BackgroundAgent
from agents.background.web_client import Page, WebClient

logger = logging.getLogger(__name__)


class ContentChunk:
    """Text chunk from a page."""

    def __init__(self, page_url: str, chunk_id: str, text: str):
        self.page_url = page_url
        self.chunk_id = chunk_id
        self.text = text


class ContentIndexerAgent(BackgroundAgent):
    """Agent for crawling and indexing website content.

    Performs BFS crawl of configured websites and generates embeddings
    for semantic search using Ollama.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize content indexer agent."""
        if config_path is None:
            config_path = Path("config/website_monitoring_agents.toml")

        super().__init__(config_path, verbose)

        self.websites = self.config.tasks.get("websites", [])
        self.rate_limit_seconds = self.config.tasks.get("rate_limit_seconds", 1.0)
        self.max_pages_per_site = self.config.tasks.get("max_pages_per_site", 500)
        self.chunk_size = self.config.tasks.get("chunk_size", 512)
        self.chunk_overlap = self.config.tasks.get("chunk_overlap", 50)

        # Embedding configuration
        self.embedding_model = self.config.tasks.get("embedding_model", "nomic-embed-text")
        self.ollama_url = self.config.tasks.get(
            "ollama_url", "http://100.104.53.21:11434"
        )
        self.batch_size = self.config.tasks.get("batch_size", 50)

    def _default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path("config/website_monitoring_agents.toml")

    def run(self) -> dict[str, Any]:
        """Execute content indexing for all configured websites.

        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Indexing {len(self.websites)} websites")

        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "websites_indexed": len(self.websites),
            "websites": [],
            "summary": {},
        }

        total_pages = 0
        total_chunks = 0
        total_embeddings = 0
        total_errors = 0

        # Index each website
        for site in self.websites:
            site_name = site.get("name", "unknown")
            logger.info(f"Processing {site_name}...")

            try:
                site_result = asyncio.run(self._index_website(site))
                results["websites"].append(site_result)

                total_pages += site_result["pages_crawled"]
                total_chunks += site_result["chunks_created"]
                total_embeddings += site_result["embeddings_generated"]
                total_errors += site_result["errors"]

            except Exception as e:
                logger.error(f"Failed to index {site_name}: {e}")
                total_errors += 1
                results["websites"].append(
                    {
                        "site": site_name,
                        "error": str(e),
                        "pages_crawled": 0,
                        "chunks_created": 0,
                        "embeddings_generated": 0,
                    }
                )

        # Summary
        results["summary"] = {
            "total_pages_crawled": total_pages,
            "total_chunks_created": total_chunks,
            "total_embeddings_generated": total_embeddings,
            "total_errors": total_errors,
        }

        summary_text = (
            f"Indexed {len(self.websites)} websites: "
            f"{total_pages} pages, {total_chunks} chunks, "
            f"{total_embeddings} embeddings, {total_errors} errors"
        )
        results["summary"]["text"] = summary_text

        # Save results
        self._save_output(results, "content_index")

        logger.info(summary_text)

        return results

    async def _index_website(self, site_config: dict) -> dict[str, Any]:
        """Index a single website.

        Args:
            site_config: Website configuration

        Returns:
            Dictionary with indexing results
        """
        site_name = site_config.get("name", "unknown")
        url = site_config.get("url")
        max_depth = site_config.get("max_depth", 3)

        logger.info(f"Crawling {site_name}: {url}")

        # Crawl website
        async with WebClient(
            url,
            rate_limit=self.rate_limit_seconds,
            user_agent=self.config.tasks.get("user_agent", "hafs-indexer/1.0"),
        ) as client:
            pages = await client.bfs_crawl(
                max_pages=self.max_pages_per_site,
                max_depth=max_depth,
            )

        logger.info(f"Crawled {len(pages)} pages from {site_name}")

        # Chunk content
        chunks = []
        for page in pages:
            page_chunks = self._chunk_page(page)
            chunks.extend(page_chunks)

        logger.info(f"Created {len(chunks)} chunks from {site_name}")

        # Generate embeddings
        kb_dir = (
            Path(self.config.tasks.get("output_dir", "~/.context/knowledge/websites"))
            .expanduser()
            / site_name.replace(".", "-")
        )
        kb_dir.mkdir(parents=True, exist_ok=True)

        embeddings_generated, errors = await self._generate_embeddings(chunks, kb_dir)

        # Save page metadata
        self._save_page_metadata(pages, kb_dir)

        # Save crawl history
        self._save_crawl_history(site_name, len(pages), len(chunks), kb_dir)

        return {
            "site": site_name,
            "url": url,
            "crawl_timestamp": datetime.now().isoformat(),
            "pages_crawled": len(pages),
            "chunks_created": len(chunks),
            "embeddings_generated": embeddings_generated,
            "errors": errors,
            "knowledge_base_path": str(kb_dir),
        }

    def _chunk_page(self, page: Page) -> list[ContentChunk]:
        """Chunk page content into smaller segments.

        Args:
            page: Page object

        Returns:
            List of content chunks
        """
        text = page.text
        chunks = []

        # Simple chunking with overlap
        chunk_id = 0
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i : i + self.chunk_size]

            # Skip very small chunks
            if len(chunk_text) < 50:
                continue

            chunk = ContentChunk(
                page_url=page.url,
                chunk_id=f"{self._url_to_id(page.url)}_{chunk_id}",
                text=chunk_text,
            )
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _url_to_id(self, url: str) -> str:
        """Convert URL to a safe ID.

        Args:
            url: URL string

        Returns:
            Safe ID string
        """
        return hashlib.md5(url.encode()).hexdigest()[:12]

    async def _generate_embeddings(
        self, chunks: list[ContentChunk], kb_dir: Path
    ) -> tuple[int, int]:
        """Generate embeddings for chunks using Ollama.

        Args:
            chunks: List of content chunks
            kb_dir: Knowledge base directory

        Returns:
            Tuple of (embeddings_generated, errors)
        """
        embeddings_dir = kb_dir / "embeddings" / self.embedding_model.replace(":", "-")
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load index
        index_file = kb_dir / f"embedding_index_{self.embedding_model.replace(':', '-')}.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
        else:
            index = {}

        embeddings_generated = 0
        errors = 0

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            logger.info(
                f"Processing embedding batch {i // self.batch_size + 1} "
                f"({len(batch)} chunks)"
            )

            for chunk in batch:
                # Skip if already embedded
                if chunk.chunk_id in index:
                    continue

                try:
                    # Generate embedding
                    embedding = await self._embed_text(chunk.text)

                    if embedding:
                        # Save embedding
                        emb_hash = hashlib.md5(chunk.chunk_id.encode()).hexdigest()
                        emb_file = embeddings_dir / f"{emb_hash}.json"

                        with open(emb_file, "w") as f:
                            json.dump(
                                {
                                    "id": chunk.chunk_id,
                                    "page_url": chunk.page_url,
                                    "text_preview": chunk.text[:200],
                                    "embedding": embedding,
                                    "timestamp": datetime.now().isoformat(),
                                },
                                f,
                                indent=2,
                            )

                        # Update index
                        index[chunk.chunk_id] = f"{emb_hash}.json"
                        embeddings_generated += 1

                except Exception as e:
                    logger.error(f"Failed to generate embedding for {chunk.chunk_id}: {e}")
                    errors += 1

            # Save index after each batch
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)

            # Small delay between batches
            await asyncio.sleep(0.5)

        logger.info(f"Generated {embeddings_generated} embeddings ({errors} errors)")

        return embeddings_generated, errors

    async def _embed_text(self, text: str) -> Optional[list[float]]:
        """Generate embedding for text using Ollama API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on error
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("embedding")
                else:
                    logger.error(
                        f"Embedding request failed: {response.status_code} - {response.text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _save_page_metadata(self, pages: list[Page], kb_dir: Path):
        """Save page metadata to knowledge base.

        Args:
            pages: List of crawled pages
            kb_dir: Knowledge base directory
        """
        pages_file = kb_dir / "pages.json"

        pages_data = []
        for page in pages:
            pages_data.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "status_code": page.status_code,
                    "crawl_time_ms": page.crawl_time_ms,
                    "num_links": len(page.links),
                    "text_length": len(page.text),
                }
            )

        with open(pages_file, "w") as f:
            json.dump(pages_data, f, indent=2)

        logger.debug(f"Saved metadata for {len(pages)} pages")

    def _save_crawl_history(
        self, site_name: str, pages_count: int, chunks_count: int, kb_dir: Path
    ):
        """Save crawl history.

        Args:
            site_name: Website name
            pages_count: Number of pages crawled
            chunks_count: Number of chunks created
            kb_dir: Knowledge base directory
        """
        history_file = kb_dir / "crawl_history.json"

        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Add new entry
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "site": site_name,
                "pages_crawled": pages_count,
                "chunks_created": chunks_count,
            }
        )

        # Keep last 30 entries
        history = history[-30:]

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)


def main():
    """CLI entry point for content indexer agent."""
    parser = argparse.ArgumentParser(description="hafs Content Indexer Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = ContentIndexerAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
