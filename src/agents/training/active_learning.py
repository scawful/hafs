"""Active Learning Sampler for training data pipeline.

Targets generation at embedding space gaps to improve coverage:
- Identifies sparse regions in embedding space
- Prioritizes sample generation in under-represented areas
- Balances exploration vs exploitation
- Tracks coverage metrics over time
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable, Awaitable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRegion:
    """A region in embedding space."""

    centroid: NDArray[np.float32]
    sample_count: int
    sample_ids: list[str] = field(default_factory=list)
    domain: str = "unknown"
    avg_quality: float = 0.0

    @property
    def density(self) -> float:
        """Relative density of this region."""
        return self.sample_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "centroid": self.centroid.tolist(),
            "sample_count": self.sample_count,
            "sample_ids": self.sample_ids[-100:],  # Keep last 100
            "domain": self.domain,
            "avg_quality": self.avg_quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingRegion":
        """Deserialize from dictionary."""
        return cls(
            centroid=np.array(data["centroid"], dtype=np.float32),
            sample_count=data["sample_count"],
            sample_ids=data.get("sample_ids", []),
            domain=data.get("domain", "unknown"),
            avg_quality=data.get("avg_quality", 0.0),
        )


@dataclass
class CoverageReport:
    """Report on embedding space coverage."""

    total_samples: int
    num_regions: int
    avg_region_density: float
    min_region_density: int
    max_region_density: int
    sparse_regions: int  # Regions with below-average density
    coverage_score: float  # 0-1, higher means better coverage
    domain_coverage: dict[str, float] = field(default_factory=dict)


class ActiveLearningSampler:
    """Samples embedding space to identify gaps and prioritize generation."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_regions: int = 50,
        storage_path: Optional[Path] = None,
    ):
        """Initialize active learning sampler.

        Args:
            embedding_dim: Dimension of embeddings
            num_regions: Number of regions to partition embedding space into
            storage_path: Path to store sampler state
        """
        self.embedding_dim = embedding_dim
        self.num_regions = num_regions
        self.storage_path = storage_path or Path.home() / ".context" / "training" / "active_learning.json"

        # Regions in embedding space
        self.regions: list[EmbeddingRegion] = []

        # All embeddings for computing regions
        self._embeddings: list[NDArray[np.float32]] = []
        self._sample_ids: list[str] = []
        self._domains: list[str] = []

        # Generation priorities
        self._region_priorities: NDArray[np.float32] = np.array([])

        self._load()

    def _load(self) -> None:
        """Load existing sampler state."""
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())

            for region_data in data.get("regions", []):
                self.regions.append(EmbeddingRegion.from_dict(region_data))

            if self.regions:
                self._compute_priorities()

            logger.info(f"Loaded {len(self.regions)} regions from {self.storage_path}")

        except Exception as e:
            logger.warning(f"Failed to load active learning state: {e}")

    def save(self) -> None:
        """Save sampler state to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "regions": [r.to_dict() for r in self.regions],
            "embedding_dim": self.embedding_dim,
            "num_regions": self.num_regions,
            "last_updated": datetime.now().isoformat(),
        }

        self.storage_path.write_text(json.dumps(data, indent=2))

    def add_sample(
        self,
        sample_id: str,
        embedding: NDArray[np.float32],
        domain: str,
        quality_score: float = 0.0,
    ) -> None:
        """Add a sample to the space.

        Args:
            sample_id: Unique sample identifier
            embedding: Sample embedding vector
            domain: Domain of the sample
            quality_score: Quality score of the sample
        """
        self._embeddings.append(embedding)
        self._sample_ids.append(sample_id)
        self._domains.append(domain)

        # Update region if we have them
        if self.regions:
            region_idx = self._find_nearest_region(embedding)
            if region_idx >= 0:
                region = self.regions[region_idx]
                region.sample_count += 1
                region.sample_ids.append(sample_id)
                region.domain = domain if region.sample_count == 1 else region.domain

                # Update running average of quality
                n = region.sample_count
                region.avg_quality = ((n - 1) * region.avg_quality + quality_score) / n

                # Recompute priorities
                self._compute_priorities()

    def _find_nearest_region(self, embedding: NDArray[np.float32]) -> int:
        """Find the index of the nearest region to an embedding."""
        if not self.regions:
            return -1

        centroids = np.array([r.centroid for r in self.regions])
        distances = np.linalg.norm(centroids - embedding, axis=1)
        return int(np.argmin(distances))

    def compute_regions(self, min_samples: int = 100) -> None:
        """Compute regions using k-means clustering.

        Args:
            min_samples: Minimum samples needed before computing regions
        """
        if len(self._embeddings) < min_samples:
            logger.info(f"Need {min_samples} samples to compute regions, have {len(self._embeddings)}")
            return

        embeddings = np.array(self._embeddings)

        # Simple k-means clustering
        centroids = self._kmeans(embeddings, k=self.num_regions)

        # Create regions
        self.regions = []
        for centroid in centroids:
            self.regions.append(EmbeddingRegion(
                centroid=centroid,
                sample_count=0,
            ))

        # Assign samples to regions
        for i, embedding in enumerate(self._embeddings):
            region_idx = self._find_nearest_region(embedding)
            if region_idx >= 0:
                region = self.regions[region_idx]
                region.sample_count += 1
                region.sample_ids.append(self._sample_ids[i])
                region.domain = self._domains[i]

        self._compute_priorities()
        self.save()

        logger.info(f"Computed {len(self.regions)} regions from {len(self._embeddings)} samples")

    def _kmeans(
        self,
        embeddings: NDArray[np.float32],
        k: int,
        max_iters: int = 50,
    ) -> NDArray[np.float32]:
        """Simple k-means clustering.

        Args:
            embeddings: Array of embeddings
            k: Number of clusters
            max_iters: Maximum iterations

        Returns:
            Array of cluster centroids
        """
        n = len(embeddings)
        if n < k:
            # Not enough samples, return random subset
            return embeddings[:k].copy()

        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = embeddings[indices].copy()

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.linalg.norm(embeddings[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centroids[j] = embeddings[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids

    def _compute_priorities(self) -> None:
        """Compute generation priorities for each region.

        Lower density = higher priority (inverse relationship)
        """
        if not self.regions:
            self._region_priorities = np.array([])
            return

        densities = np.array([r.sample_count for r in self.regions], dtype=np.float32)

        # Avoid division by zero
        densities = densities + 1.0

        # Inverse density as priority
        priorities = 1.0 / densities

        # Normalize to probabilities
        self._region_priorities = priorities / priorities.sum()

    def get_sparse_regions(self, top_k: int = 10) -> list[tuple[int, EmbeddingRegion]]:
        """Get the most sparse (under-represented) regions.

        Args:
            top_k: Number of sparse regions to return

        Returns:
            List of (index, region) tuples, sorted by sparsity
        """
        if not self.regions:
            return []

        indexed = list(enumerate(self.regions))
        sorted_regions = sorted(indexed, key=lambda x: x[1].sample_count)

        return sorted_regions[:top_k]

    def sample_region_for_generation(self) -> Optional[EmbeddingRegion]:
        """Sample a region to target for next generation.

        Uses priority weighting to favor sparse regions.

        Returns:
            Region to target, or None if no regions exist
        """
        if not self.regions or len(self._region_priorities) == 0:
            return None

        # Sample proportionally to priority
        idx = np.random.choice(len(self.regions), p=self._region_priorities)
        return self.regions[idx]

    def get_target_embedding(self, region: Optional[EmbeddingRegion] = None) -> Optional[NDArray[np.float32]]:
        """Get a target embedding to guide generation.

        Args:
            region: Optional specific region to target

        Returns:
            Target embedding vector, or None
        """
        if region is None:
            region = self.sample_region_for_generation()

        if region is None:
            return None

        # Add small random noise to centroid for diversity
        noise = np.random.randn(self.embedding_dim).astype(np.float32) * 0.1
        return region.centroid + noise

    def get_coverage_report(self) -> CoverageReport:
        """Generate a coverage report.

        Returns:
            CoverageReport with coverage statistics
        """
        if not self.regions:
            return CoverageReport(
                total_samples=len(self._sample_ids),
                num_regions=0,
                avg_region_density=0.0,
                min_region_density=0,
                max_region_density=0,
                sparse_regions=0,
                coverage_score=0.0,
            )

        densities = [r.sample_count for r in self.regions]
        avg_density = sum(densities) / len(densities) if densities else 0

        # Count sparse regions (below average)
        sparse_count = sum(1 for d in densities if d < avg_density)

        # Coverage score: how evenly distributed are samples?
        # Using coefficient of variation (lower = more even)
        if avg_density > 0:
            std_dev = (sum((d - avg_density) ** 2 for d in densities) / len(densities)) ** 0.5
            cv = std_dev / avg_density
            # Convert to 0-1 score (lower cv = higher score)
            coverage_score = max(0.0, min(1.0, 1.0 - cv))
        else:
            coverage_score = 0.0

        # Domain coverage
        domain_samples: dict[str, int] = {}
        for region in self.regions:
            if region.domain not in domain_samples:
                domain_samples[region.domain] = 0
            domain_samples[region.domain] += region.sample_count

        total = sum(domain_samples.values())
        domain_coverage = {
            d: count / total if total > 0 else 0.0
            for d, count in domain_samples.items()
        }

        return CoverageReport(
            total_samples=sum(densities),
            num_regions=len(self.regions),
            avg_region_density=avg_density,
            min_region_density=min(densities) if densities else 0,
            max_region_density=max(densities) if densities else 0,
            sparse_regions=sparse_count,
            coverage_score=coverage_score,
            domain_coverage=domain_coverage,
        )

    async def suggest_prompts_for_sparse_regions(
        self,
        prompt_generator: Callable[[NDArray[np.float32], str], Awaitable[str]],
        top_k: int = 5,
    ) -> list[str]:
        """Generate prompts targeting sparse regions.

        Args:
            prompt_generator: Async function that generates prompts given target embedding and domain
            top_k: Number of prompts to generate

        Returns:
            List of generated prompts
        """
        sparse_regions = self.get_sparse_regions(top_k)

        prompts = []
        for idx, region in sparse_regions:
            target_embedding = self.get_target_embedding(region)
            if target_embedding is not None:
                try:
                    prompt = await prompt_generator(target_embedding, region.domain)
                    prompts.append(prompt)
                except Exception as e:
                    logger.warning(f"Failed to generate prompt for region {idx}: {e}")

        return prompts

    def get_diversity_score(self, embedding: NDArray[np.float32]) -> float:
        """Score how diverse a new sample would be.

        Args:
            embedding: Embedding of potential new sample

        Returns:
            Diversity score 0-1, higher means more novel
        """
        if not self.regions:
            return 1.0  # Maximally diverse if no regions yet

        region_idx = self._find_nearest_region(embedding)
        if region_idx < 0:
            return 1.0

        region = self.regions[region_idx]

        # Lower density = higher diversity
        max_density = max(r.sample_count for r in self.regions)
        if max_density == 0:
            return 1.0

        diversity = 1.0 - (region.sample_count / max_density)
        return max(0.0, min(1.0, diversity))

    def export_visualization_data(self) -> dict[str, Any]:
        """Export data for visualization.

        Returns:
            Data suitable for plotting (e.g., t-SNE visualization)
        """
        if not self.regions:
            return {"regions": [], "priorities": []}

        return {
            "regions": [
                {
                    "index": i,
                    "centroid_norm": float(np.linalg.norm(r.centroid)),
                    "sample_count": r.sample_count,
                    "domain": r.domain,
                    "avg_quality": r.avg_quality,
                }
                for i, r in enumerate(self.regions)
            ],
            "priorities": self._region_priorities.tolist() if len(self._region_priorities) > 0 else [],
            "total_samples": sum(r.sample_count for r in self.regions),
            "coverage": self.get_coverage_report().coverage_score,
        }
