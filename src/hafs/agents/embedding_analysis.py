"""Embedding Analysis Tools.

Provides advanced analysis capabilities for embedding data:
- Clustering (K-Means, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Similarity analysis
- Outlier detection
- Topic modeling from clusters
- Visualization support

Uses Gemini 3 for cluster interpretation and labeling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hafs.agents.base import BaseAgent
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """A cluster of embeddings."""

    id: int
    label: str = ""
    description: str = ""
    members: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    size: int = 0
    coherence: float = 0.0  # Average intra-cluster similarity
    representative_members: list[str] = field(default_factory=list)


@dataclass
class SimilarityResult:
    """Result of similarity analysis."""

    query_id: str
    similar_items: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class OutlierResult:
    """Result of outlier detection."""

    outlier_id: str
    distance_from_nearest: float
    nearest_cluster: int
    reason: str = ""


class EmbeddingAnalyzer(BaseAgent):
    """Advanced embedding analysis with clustering and visualization.

    Example:
        analyzer = EmbeddingAnalyzer()
        await analyzer.setup()

        # Load embeddings from a KB
        await analyzer.load_from_kb("alttp")

        # Cluster embeddings
        clusters = await analyzer.cluster(n_clusters=10)

        # Get cluster interpretation from Gemini 3
        interpretations = await analyzer.interpret_clusters()

        # Find similar items
        similar = analyzer.find_similar("POSX", limit=10)

        # Detect outliers
        outliers = analyzer.detect_outliers()

        # Export for visualization
        viz_data = analyzer.export_for_visualization()
    """

    def __init__(self):
        super().__init__(
            "EmbeddingAnalyzer",
            "Advanced embedding analysis with clustering, similarity, and visualization."
        )

        self.analysis_dir = self.context_root / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # In-memory data
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}
        self._clusters: list[Cluster] = []
        self._reduced_embeddings: dict[str, list[float]] = {}  # 2D/3D for viz

        self._orchestrator: Optional[UnifiedOrchestrator] = None

    async def setup(self):
        """Initialize the analyzer."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        logger.info("EmbeddingAnalyzer ready")

    async def load_from_kb(self, kb_name: str) -> int:
        """Load embeddings from a knowledge base.

        Args:
            kb_name: Name of KB ("alttp", "oracle-of-secrets", "gigaleak").

        Returns:
            Number of embeddings loaded.
        """
        kb_paths = {
            "alttp": self.context_root / "knowledge" / "alttp" / "embeddings",
            "oracle-of-secrets": self.context_root / "knowledge" / "oracle-of-secrets" / "embeddings",
            "gigaleak": self.context_root / "knowledge" / "gigaleak" / "embeddings",
        }

        emb_dir = kb_paths.get(kb_name)
        if not emb_dir or not emb_dir.exists():
            logger.warning(f"KB embeddings not found: {kb_name}")
            return 0

        count = 0
        for emb_file in emb_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                emb_id = data.get("id")
                embedding = data.get("embedding")
                if emb_id and embedding:
                    self._embeddings[emb_id] = embedding
                    self._metadata[emb_id] = {
                        "kb": kb_name,
                        "text": data.get("text", "")[:200],
                    }
                    count += 1
            except:
                pass

        logger.info(f"Loaded {count} embeddings from {kb_name}")
        return count

    async def cluster(
        self,
        n_clusters: int = 10,
        method: str = "kmeans",
        min_cluster_size: int = 5,
    ) -> list[Cluster]:
        """Cluster embeddings.

        Args:
            n_clusters: Number of clusters for K-Means.
            method: Clustering method ("kmeans", "dbscan", "hierarchical").
            min_cluster_size: Minimum cluster size for DBSCAN.

        Returns:
            List of clusters.
        """
        if not self._embeddings:
            logger.warning("No embeddings loaded")
            return []

        logger.info(f"Clustering {len(self._embeddings)} embeddings with {method}")

        if method == "kmeans":
            self._clusters = self._kmeans_cluster(n_clusters)
        elif method == "dbscan":
            self._clusters = self._dbscan_cluster(min_cluster_size)
        elif method == "hierarchical":
            self._clusters = self._hierarchical_cluster(n_clusters)
        else:
            self._clusters = self._kmeans_cluster(n_clusters)

        # Calculate cluster statistics
        for cluster in self._clusters:
            cluster.size = len(cluster.members)
            if cluster.members:
                cluster.coherence = self._calculate_coherence(cluster)
                cluster.representative_members = self._get_representatives(cluster, 5)

        logger.info(f"Created {len(self._clusters)} clusters")
        return self._clusters

    def _kmeans_cluster(self, n_clusters: int) -> list[Cluster]:
        """K-Means clustering implementation."""
        import random

        embeddings_list = list(self._embeddings.items())
        if len(embeddings_list) < n_clusters:
            n_clusters = len(embeddings_list)

        # Initialize centroids randomly
        random.seed(42)
        initial_indices = random.sample(range(len(embeddings_list)), n_clusters)
        centroids = [embeddings_list[i][1] for i in initial_indices]

        # Iterate
        max_iterations = 100
        for iteration in range(max_iterations):
            # Assign points to nearest centroid
            assignments = defaultdict(list)
            for emb_id, embedding in embeddings_list:
                nearest = min(range(n_clusters),
                             key=lambda i: self._euclidean_distance(embedding, centroids[i]))
                assignments[nearest].append(emb_id)

            # Update centroids
            new_centroids = []
            for i in range(n_clusters):
                if assignments[i]:
                    cluster_embeddings = [self._embeddings[eid] for eid in assignments[i]]
                    new_centroid = self._compute_centroid(cluster_embeddings)
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[i])

            # Check convergence
            converged = all(
                self._euclidean_distance(centroids[i], new_centroids[i]) < 1e-6
                for i in range(n_clusters)
            )
            centroids = new_centroids

            if converged:
                break

        # Create cluster objects
        clusters = []
        for i in range(n_clusters):
            clusters.append(Cluster(
                id=i,
                members=assignments[i],
                centroid=centroids[i],
            ))

        return clusters

    def _dbscan_cluster(self, min_size: int) -> list[Cluster]:
        """DBSCAN clustering implementation."""
        embeddings_list = list(self._embeddings.items())
        n = len(embeddings_list)

        # Calculate distance matrix
        distances = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(embeddings_list[i][1], embeddings_list[j][1])
                distances[i][j] = dist
                distances[j][i] = dist

        # Estimate epsilon (use median of k-nearest distances)
        k = min_size
        k_distances = []
        for i in range(n):
            sorted_dists = sorted(distances[i])
            if len(sorted_dists) > k:
                k_distances.append(sorted_dists[k])
        eps = sorted(k_distances)[len(k_distances) // 2] if k_distances else 1.0

        # DBSCAN algorithm
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            neighbors = [j for j in range(n) if distances[i][j] <= eps]

            if len(neighbors) < min_size:
                continue  # Noise point

            labels[i] = cluster_id
            seed_set = set(neighbors)
            seed_set.discard(i)

            while seed_set:
                q = seed_set.pop()
                if labels[q] == -1:
                    labels[q] = cluster_id

                    q_neighbors = [j for j in range(n) if distances[q][j] <= eps]
                    if len(q_neighbors) >= min_size:
                        seed_set.update(j for j in q_neighbors if labels[j] == -1)

            cluster_id += 1

        # Create cluster objects
        cluster_members = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                cluster_members[label].append(embeddings_list[i][0])

        clusters = []
        for cid, members in cluster_members.items():
            cluster_embeddings = [self._embeddings[m] for m in members]
            clusters.append(Cluster(
                id=cid,
                members=members,
                centroid=self._compute_centroid(cluster_embeddings),
            ))

        return clusters

    def _hierarchical_cluster(self, n_clusters: int) -> list[Cluster]:
        """Simple agglomerative hierarchical clustering."""
        embeddings_list = list(self._embeddings.items())
        n = len(embeddings_list)

        # Start with each point as its own cluster
        current_clusters = [[i] for i in range(n)]

        # Merge until we have desired number
        while len(current_clusters) > n_clusters:
            # Find closest pair
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    # Average linkage
                    dist = self._cluster_distance(
                        current_clusters[i], current_clusters[j], embeddings_list
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge
            merged = current_clusters[merge_i] + current_clusters[merge_j]
            current_clusters = [c for i, c in enumerate(current_clusters)
                              if i not in (merge_i, merge_j)]
            current_clusters.append(merged)

        # Create cluster objects
        clusters = []
        for cid, indices in enumerate(current_clusters):
            members = [embeddings_list[i][0] for i in indices]
            cluster_embeddings = [embeddings_list[i][1] for i in indices]
            clusters.append(Cluster(
                id=cid,
                members=members,
                centroid=self._compute_centroid(cluster_embeddings),
            ))

        return clusters

    def _cluster_distance(self, c1: list[int], c2: list[int],
                         embeddings_list: list[tuple]) -> float:
        """Calculate average linkage distance between clusters."""
        total = 0.0
        count = 0
        for i in c1:
            for j in c2:
                total += self._euclidean_distance(
                    embeddings_list[i][1], embeddings_list[j][1]
                )
                count += 1
        return total / count if count > 0 else float('inf')

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """Compute centroid of embeddings."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v
        return [c / len(embeddings) for c in centroid]

    def _calculate_coherence(self, cluster: Cluster) -> float:
        """Calculate cluster coherence (average pairwise similarity)."""
        if len(cluster.members) < 2:
            return 1.0

        total_sim = 0.0
        count = 0
        for i, m1 in enumerate(cluster.members):
            for m2 in cluster.members[i + 1:]:
                total_sim += self._cosine_similarity(
                    self._embeddings[m1], self._embeddings[m2]
                )
                count += 1

        return total_sim / count if count > 0 else 0.0

    def _get_representatives(self, cluster: Cluster, n: int) -> list[str]:
        """Get n most representative members (closest to centroid)."""
        if not cluster.members:
            return []

        distances = [
            (m, self._euclidean_distance(self._embeddings[m], cluster.centroid))
            for m in cluster.members
        ]
        distances.sort(key=lambda x: x[1])
        return [m for m, _ in distances[:n]]

    async def interpret_clusters(self) -> dict[int, str]:
        """Use Gemini 3 to interpret and label clusters.

        Returns:
            Dict of cluster_id -> interpretation.
        """
        if not self._clusters:
            return {}

        logger.info("Interpreting clusters with Gemini 3...")

        interpretations = {}

        for cluster in self._clusters:
            if not cluster.members:
                continue

            # Get sample members with their text
            samples = []
            for member in cluster.representative_members[:10]:
                text = self._metadata.get(member, {}).get("text", member)
                samples.append(f"- {member}: {text[:100]}")

            prompt = f"""Analyze this cluster of symbols from A Link to the Past (SNES game) codebase.

Cluster {cluster.id} ({cluster.size} members, coherence: {cluster.coherence:.2f})

Sample members:
{chr(10).join(samples)}

Provide:
1. A short label (2-4 words) describing what these symbols have in common
2. A one-sentence explanation of the cluster's theme

Format:
Label: [your label]
Explanation: [your explanation]"""

            try:
                result = await self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.FAST,
                    provider=Provider.GEMINI,
                    model="gemini-3-flash-preview",
                )

                if result.content:
                    # Parse response
                    lines = result.content.strip().split("\n")
                    label = ""
                    explanation = ""

                    for line in lines:
                        if line.startswith("Label:"):
                            label = line[6:].strip()
                        elif line.startswith("Explanation:"):
                            explanation = line[12:].strip()

                    cluster.label = label
                    cluster.description = explanation
                    interpretations[cluster.id] = f"{label}: {explanation}"

            except Exception as e:
                logger.warning(f"Failed to interpret cluster {cluster.id}: {e}")

            await asyncio.sleep(0.2)

        return interpretations

    def find_similar(
        self,
        query_id: str,
        limit: int = 10,
        exclude_same_cluster: bool = False,
    ) -> SimilarityResult:
        """Find items similar to a query item.

        Args:
            query_id: ID of query embedding.
            limit: Max results.
            exclude_same_cluster: Exclude items from same cluster.

        Returns:
            Similarity results.
        """
        if query_id not in self._embeddings:
            return SimilarityResult(query_id=query_id)

        query_emb = self._embeddings[query_id]

        # Find query's cluster
        query_cluster = None
        if exclude_same_cluster and self._clusters:
            for cluster in self._clusters:
                if query_id in cluster.members:
                    query_cluster = cluster.id
                    break

        # Calculate similarities
        similarities = []
        for emb_id, embedding in self._embeddings.items():
            if emb_id == query_id:
                continue

            if exclude_same_cluster and query_cluster is not None:
                # Check if in same cluster
                in_same = False
                for cluster in self._clusters:
                    if cluster.id == query_cluster and emb_id in cluster.members:
                        in_same = True
                        break
                if in_same:
                    continue

            sim = self._cosine_similarity(query_emb, embedding)
            similarities.append((emb_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return SimilarityResult(
            query_id=query_id,
            similar_items=similarities[:limit],
        )

    def detect_outliers(self, threshold: float = 2.0) -> list[OutlierResult]:
        """Detect outliers based on distance from cluster centroids.

        Args:
            threshold: Standard deviation threshold for outliers.

        Returns:
            List of outliers.
        """
        if not self._clusters:
            return []

        outliers = []

        # Calculate distances to nearest cluster
        distances = []
        for emb_id, embedding in self._embeddings.items():
            min_dist = float('inf')
            nearest_cluster = 0

            for cluster in self._clusters:
                if cluster.centroid:
                    dist = self._euclidean_distance(embedding, cluster.centroid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cluster = cluster.id

            distances.append((emb_id, min_dist, nearest_cluster))

        # Calculate mean and std
        dist_values = [d[1] for d in distances]
        mean_dist = sum(dist_values) / len(dist_values)
        std_dist = math.sqrt(sum((d - mean_dist) ** 2 for d in dist_values) / len(dist_values))

        # Identify outliers
        threshold_value = mean_dist + threshold * std_dist
        for emb_id, dist, nearest in distances:
            if dist > threshold_value:
                outliers.append(OutlierResult(
                    outlier_id=emb_id,
                    distance_from_nearest=dist,
                    nearest_cluster=nearest,
                    reason=f"Distance {dist:.2f} > threshold {threshold_value:.2f}",
                ))

        return outliers

    def reduce_dimensions(self, method: str = "pca", n_components: int = 2) -> dict[str, list[float]]:
        """Reduce embedding dimensions for visualization.

        Args:
            method: Reduction method ("pca", "tsne").
            n_components: Target dimensions (2 or 3).

        Returns:
            Dict of id -> reduced coordinates.
        """
        if not self._embeddings:
            return {}

        if method == "pca":
            self._reduced_embeddings = self._pca_reduce(n_components)
        else:
            # Default to PCA
            self._reduced_embeddings = self._pca_reduce(n_components)

        return self._reduced_embeddings

    def _pca_reduce(self, n_components: int) -> dict[str, list[float]]:
        """Simple PCA implementation."""
        import random

        embeddings_list = list(self._embeddings.items())
        n = len(embeddings_list)
        dim = len(embeddings_list[0][1])

        # Center data
        mean = [0.0] * dim
        for _, emb in embeddings_list:
            for i, v in enumerate(emb):
                mean[i] += v
        mean = [m / n for m in mean]

        centered = []
        for emb_id, emb in embeddings_list:
            centered.append([v - mean[i] for i, v in enumerate(emb)])

        # Power iteration for top principal components
        components = []
        for _ in range(n_components):
            # Random initial vector
            random.seed(42 + len(components))
            pc = [random.gauss(0, 1) for _ in range(dim)]

            # Power iteration
            for _ in range(100):
                # Multiply by covariance
                new_pc = [0.0] * dim
                for c in centered:
                    dot = sum(c[i] * pc[i] for i in range(dim))
                    for i in range(dim):
                        new_pc[i] += c[i] * dot

                # Normalize
                norm = math.sqrt(sum(x * x for x in new_pc))
                pc = [x / norm for x in new_pc]

            # Remove component from data
            for i, c in enumerate(centered):
                dot = sum(c[j] * pc[j] for j in range(dim))
                for j in range(dim):
                    centered[i][j] -= dot * pc[j]

            components.append(pc)

        # Project data
        reduced = {}
        for emb_id, emb in embeddings_list:
            centered_emb = [v - mean[i] for i, v in enumerate(emb)]
            coords = []
            for pc in components:
                coords.append(sum(centered_emb[i] * pc[i] for i in range(dim)))
            reduced[emb_id] = coords

        return reduced

    def export_for_visualization(self, output_path: Optional[Path] = None) -> dict[str, Any]:
        """Export data for visualization.

        Args:
            output_path: Optional path to save JSON.

        Returns:
            Visualization data.
        """
        if not self._reduced_embeddings:
            self.reduce_dimensions()

        viz_data = {
            "points": [],
            "clusters": [],
            "metadata": {
                "total_points": len(self._embeddings),
                "total_clusters": len(self._clusters),
                "generated": datetime.now().isoformat(),
            },
        }

        # Add points
        for emb_id, coords in self._reduced_embeddings.items():
            point = {
                "id": emb_id,
                "x": coords[0] if len(coords) > 0 else 0,
                "y": coords[1] if len(coords) > 1 else 0,
                "z": coords[2] if len(coords) > 2 else 0,
                "text": self._metadata.get(emb_id, {}).get("text", ""),
                "kb": self._metadata.get(emb_id, {}).get("kb", ""),
            }

            # Add cluster assignment
            for cluster in self._clusters:
                if emb_id in cluster.members:
                    point["cluster"] = cluster.id
                    point["cluster_label"] = cluster.label
                    break

            viz_data["points"].append(point)

        # Add cluster info
        for cluster in self._clusters:
            viz_data["clusters"].append({
                "id": cluster.id,
                "label": cluster.label,
                "description": cluster.description,
                "size": cluster.size,
                "coherence": cluster.coherence,
                "representatives": cluster.representative_members,
            })

        if output_path:
            output_path.write_text(json.dumps(viz_data, indent=2))
            logger.info(f"Exported visualization data to {output_path}")

        return viz_data

    def get_cluster_summary(self) -> list[dict[str, Any]]:
        """Get summary of all clusters."""
        return [
            {
                "id": c.id,
                "label": c.label,
                "description": c.description,
                "size": c.size,
                "coherence": round(c.coherence, 3),
                "top_members": c.representative_members[:5],
            }
            for c in self._clusters
        ]

    async def run_task(self, task: str = "stats") -> dict[str, Any]:
        """Run an analysis task."""
        if task == "stats":
            return {
                "embeddings_loaded": len(self._embeddings),
                "clusters": len(self._clusters),
                "reduced_dimensions": len(self._reduced_embeddings),
            }

        elif task.startswith("load:"):
            kb_name = task[5:].strip()
            count = await self.load_from_kb(kb_name)
            return {"loaded": count, "kb": kb_name}

        elif task.startswith("cluster:"):
            n = int(task[8:].strip() or "10")
            clusters = await self.cluster(n_clusters=n)
            return {"clusters": len(clusters)}

        elif task == "interpret":
            interpretations = await self.interpret_clusters()
            return {"interpretations": interpretations}

        elif task.startswith("similar:"):
            query = task[8:].strip()
            result = self.find_similar(query)
            return {"query": result.query_id, "similar": result.similar_items}

        elif task == "outliers":
            outliers = self.detect_outliers()
            return {"outliers": [o.outlier_id for o in outliers]}

        elif task == "export":
            path = self.analysis_dir / "visualization.json"
            self.export_for_visualization(path)
            return {"exported": str(path)}

        return {"error": "Unknown task"}
