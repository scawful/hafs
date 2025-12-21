"""Cataloger agent - organizes training datasets and models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.background.base import BackgroundAgent


class CatalogerAgent(BackgroundAgent):
    """Cataloger agent for dataset and model organization.

    Indexes datasets, tracks model checkpoints, generates metadata,
    and provides storage optimization recommendations.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize cataloger agent."""
        super().__init__(config_path, verbose)
        self.scan_dirs = self.config.tasks.get("scan_directories", [])

    def run(self) -> dict[str, Any]:
        """Execute cataloging tasks.

        Returns:
            Dictionary with cataloging results
        """
        results = {
            "datasets": [],
            "models": [],
            "storage_usage": {},
            "recommendations": [],
        }

        for scan_dir in self.scan_dirs:
            dir_path = Path(scan_dir)
            if not dir_path.exists():
                self.logger.warning(f"Directory not found: {scan_dir}")
                continue

            self.logger.info(f"Cataloging directory: {scan_dir}")

            # Catalog datasets
            if "dataset" in scan_dir.lower():
                datasets = self._catalog_datasets(dir_path)
                results["datasets"].extend(datasets)

            # Catalog models
            if "model" in scan_dir.lower():
                models = self._catalog_models(dir_path)
                results["models"].extend(models)

            # Calculate storage usage
            usage = self._calculate_storage(dir_path)
            results["storage_usage"][scan_dir] = usage

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Save results
        self._save_output(results, "catalog_report")

        return results

    def _catalog_datasets(self, path: Path) -> list[dict[str, Any]]:
        """Catalog training datasets.

        Args:
            path: Directory containing datasets

        Returns:
            List of dataset metadata
        """
        datasets = []

        for file_path in path.glob("*.jsonl"):
            try:
                # Get file stats
                stats = file_path.stat()
                size_mb = stats.st_size / (1024 * 1024)

                # Count samples
                with open(file_path) as f:
                    sample_count = sum(1 for _ in f)

                # Calculate checksum
                checksum = self._calculate_checksum(file_path)

                dataset_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(size_mb, 2),
                    "sample_count": sample_count,
                    "checksum": checksum,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                }

                # Parse metadata from filename
                # Expected format: alttp_yaze_full_34500_asm.jsonl
                name_parts = file_path.stem.split("_")
                if len(name_parts) >= 4:
                    dataset_info["domain"] = name_parts[-1]  # asm, yaze, oracle, etc.
                    if name_parts[-2].isdigit():
                        dataset_info["target_samples"] = int(name_parts[-2])

                datasets.append(dataset_info)
                self.logger.info(f"Cataloged dataset: {file_path.name} ({sample_count} samples, {size_mb:.1f} MB)")

            except Exception as e:
                self.logger.warning(f"Failed to catalog {file_path}: {e}")

        return datasets

    def _catalog_models(self, path: Path) -> list[dict[str, Any]]:
        """Catalog trained models.

        Args:
            path: Directory containing models

        Returns:
            List of model metadata
        """
        models = []

        for model_dir in path.iterdir():
            if not model_dir.is_dir():
                continue

            try:
                # Look for model files
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))

                if not model_files:
                    continue

                # Get directory size
                total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                size_gb = total_size / (1024 * 1024 * 1024)

                # Look for config
                config_file = model_dir / "config.json"
                config_data = {}
                if config_file.exists():
                    with open(config_file) as f:
                        config_data = json.load(f)

                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_gb": round(size_gb, 2),
                    "num_files": len(list(model_dir.rglob("*"))),
                    "model_files": [f.name for f in model_files],
                    "modified": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                    "architecture": config_data.get("architectures", ["unknown"])[0] if config_data else "unknown",
                }

                models.append(model_info)
                self.logger.info(f"Cataloged model: {model_dir.name} ({size_gb:.1f} GB)")

            except Exception as e:
                self.logger.warning(f"Failed to catalog model {model_dir}: {e}")

        return models

    def _calculate_storage(self, path: Path) -> dict[str, Any]:
        """Calculate storage usage for a directory.

        Args:
            path: Directory path

        Returns:
            Storage usage information
        """
        try:
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            file_count = sum(1 for f in path.rglob("*") if f.is_file())

            return {
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate storage for {path}: {e}")
            return {}

    def _calculate_checksum(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate file checksum.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha256)

        Returns:
            Hexadecimal checksum string
        """
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate storage optimization recommendations.

        Args:
            results: Cataloging results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for large datasets
        for dataset in results["datasets"]:
            if dataset.get("size_mb", 0) > 500:
                recommendations.append(
                    f"Large dataset detected: {dataset['name']} ({dataset['size_mb']:.1f} MB) - "
                    f"consider archiving or compressing"
                )

        # Check for duplicate models
        model_names = [m.get("name", "") for m in results["models"]]
        if len(model_names) != len(set(model_names)):
            recommendations.append("Duplicate model names detected - review for cleanup")

        # Check total storage
        total_storage = sum(s.get("total_size_gb", 0) for s in results["storage_usage"].values())
        if total_storage > 100:
            recommendations.append(
                f"High storage usage detected ({total_storage:.1f} GB) - "
                f"consider archiving old checkpoints"
            )

        return recommendations


def main():
    """CLI entry point for cataloger agent."""
    parser = argparse.ArgumentParser(description="hafs Cataloger Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    agent = CatalogerAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
