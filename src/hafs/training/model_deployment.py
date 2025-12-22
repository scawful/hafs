"""Model Deployment System.

Handles:
- Pulling models from remote machines (Windows, cloud)
- Converting between formats (PyTorch, GGUF, ONNX)
- Deploying to serving backends (ollama, llama.cpp, vllm, halext nodes)
- Verifying model functionality
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from hafs.training.cross_platform import PathResolver, get_path_resolver
from hafs.training.model_registry import (
    ModelLocation,
    ModelMetadata,
    ModelRegistry,
    ServingBackend,
)

logger = logging.getLogger(__name__)

ConversionFormat = Literal["gguf", "onnx", "safetensors"]


@dataclass
class DeploymentStatus:
    """Status of a deployment operation."""

    model_id: str
    backend: ServingBackend
    status: Literal["pending", "pulling", "converting", "deploying", "completed", "failed"]
    progress: str
    error: Optional[str] = None
    started_at: str = ""
    completed_at: Optional[str] = None


class ModelDeployment:
    """Manages model deployment across serving backends."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        path_resolver: Optional[PathResolver] = None,
    ):
        """Initialize deployment system.

        Args:
            registry: Model registry instance
            path_resolver: Path resolver instance
        """
        self.registry = registry or ModelRegistry()
        self.path_resolver = path_resolver or get_path_resolver()

        # Local model storage
        self.local_models_dir = Path.home() / "Code" / "hafs" / "models"
        self.local_models_dir.mkdir(parents=True, exist_ok=True)

    def pull_model(
        self,
        model_id: str,
        source_location: Optional[ModelLocation] = None,
        destination: Optional[Path] = None,
    ) -> Path:
        """Pull model from remote location to local machine.

        Args:
            model_id: Model identifier
            source_location: Location to pull from (auto-detect if None)
            destination: Local destination (default: ~/Code/hafs/models/<model_id>)

        Returns:
            Path to pulled model

        Raises:
            ValueError: If model not found or not accessible
        """
        logger.info(f"Pulling model {model_id}...")

        # Get model metadata
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in registry")

        # Find source location
        if source_location is None:
            source_info = self.registry.find_best_location(model_id)
            if not source_info:
                raise ValueError(f"No accessible location found for {model_id}")
            source_location, source_path = source_info
        else:
            locations = dict(self.registry.get_available_locations(model_id))
            if source_location not in locations:
                raise ValueError(f"Model {model_id} not available at {source_location}")
            source_path = locations[source_location]

        # Set destination
        if destination is None:
            destination = self.local_models_dir / model_id

        destination.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Source: {source_location}:{source_path}")
        logger.info(f"  Destination: {destination}")

        # Try mount first (faster than SSH)
        local_mount_path = self.path_resolver.remote_to_local(source_path)
        if local_mount_path and local_mount_path.exists():
            logger.info("  Using network mount for transfer")
            self._copy_via_mount(local_mount_path, destination)
        else:
            # Fall back to SSH/SCP
            logger.info("  Using SSH/SCP for transfer")
            self._copy_via_ssh(source_location, source_path, destination)

        # Update registry
        self.registry.update_location(model_id, "mac", str(destination))

        logger.info(f"✓ Model pulled to {destination}")
        return destination

    def _copy_via_mount(self, source: Path, dest: Path) -> None:
        """Copy model via network mount.

        Args:
            source: Source path on mount
            dest: Destination path
        """
        try:
            # Copy all files
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(source, dest)

            logger.debug(f"Copied {source} -> {dest}")

        except Exception as e:
            logger.error(f"Failed to copy via mount: {e}")
            raise

    def _copy_via_ssh(
        self, remote_location: ModelLocation, remote_path: str, dest: Path
    ) -> None:
        """Copy model via SSH/SCP.

        Args:
            remote_location: Remote location name
            remote_path: Path on remote machine
            dest: Local destination
        """
        # Map location to remote host
        remote_host_map = {
            "windows": "medical-mechanica",
            "cloud": "cloud-gpu",
        }

        remote_host = remote_host_map.get(remote_location)
        if not remote_host:
            raise ValueError(f"No SSH host configured for {remote_location}")

        # Check if remote path exists
        if not self.path_resolver.check_remote_path(remote_host, remote_path):
            raise ValueError(f"Remote path not found: {remote_path}")

        # Build SCP command
        host_config = self.path_resolver.remote_hosts.get(remote_host)
        if not host_config:
            raise ValueError(f"Remote host {remote_host} not configured")

        # Format remote path for SCP
        from hafs.training.cross_platform import CrossPlatformPath

        cp_path = CrossPlatformPath(remote_path, host_config.platform)
        ssh_path = cp_path.to_ssh_path()

        source = f"{host_config}:{ssh_path}"
        scp_cmd = self.path_resolver.build_scp_command(
            source=source,
            dest=str(dest),
            remote_host=remote_host,
            recursive=True,
        )

        try:
            logger.debug(f"Running: {' '.join(scp_cmd)}")
            subprocess.run(scp_cmd, check=True, timeout=600)

        except subprocess.TimeoutExpired:
            raise RuntimeError("SCP transfer timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SCP transfer failed: {e}")

    def convert_to_gguf(
        self,
        model_id: str,
        quantization: str = "Q4_K_M",
        output_path: Optional[Path] = None,
    ) -> Path:
        """Convert PyTorch model to GGUF format for llama.cpp/ollama.

        Args:
            model_id: Model identifier
            quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
            output_path: Output path (default: model_dir/<model_id>.gguf)

        Returns:
            Path to GGUF file
        """
        logger.info(f"Converting {model_id} to GGUF ({quantization})...")

        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Get local model path
        mac_locations = dict(self.registry.get_available_locations(model_id))
        if "mac" not in mac_locations:
            raise ValueError(f"Model {model_id} not available locally - pull first")

        model_path = Path(mac_locations["mac"])

        # Set output path
        if output_path is None:
            output_path = model_path.parent / f"{model_id}-{quantization}.gguf"

        # Check for llama.cpp tools
        llama_cpp_dir = Path.home() / "Code" / "llama.cpp"
        if not llama_cpp_dir.exists():
            raise RuntimeError(
                "llama.cpp not found. Clone from https://github.com/ggerganov/llama.cpp"
            )

        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        quantize_bin = llama_cpp_dir / "llama-quantize"

        if not convert_script.exists():
            raise RuntimeError(f"Conversion script not found: {convert_script}")

        # Step 1: Convert to GGUF (F16)
        logger.info("  [1/2] Converting to F16 GGUF...")
        f16_output = model_path.parent / f"{model_id}-f16.gguf"

        convert_cmd = [
            "python",
            str(convert_script),
            str(model_path),
            "--outfile",
            str(f16_output),
            "--outtype",
            "f16",
        ]

        try:
            subprocess.run(convert_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr.decode()}")
            raise

        # Step 2: Quantize
        if quantization != "f16" and quantize_bin.exists():
            logger.info(f"  [2/2] Quantizing to {quantization}...")

            quantize_cmd = [
                str(quantize_bin),
                str(f16_output),
                str(output_path),
                quantization,
            ]

            try:
                subprocess.run(quantize_cmd, check=True, capture_output=True)
                # Remove F16 intermediate
                f16_output.unlink()
            except subprocess.CalledProcessError as e:
                logger.error(f"Quantization failed: {e.stderr.decode()}")
                raise
        else:
            output_path = f16_output

        logger.info(f"✓ GGUF model saved to {output_path}")

        # Update registry
        model.formats.append("gguf")
        self.registry._save_registry()

        return output_path

    def deploy_to_ollama(
        self,
        model_id: str,
        ollama_model_name: Optional[str] = None,
        quantization: str = "Q4_K_M",
    ) -> str:
        """Deploy model to Ollama.

        Args:
            model_id: Model identifier
            ollama_model_name: Name in Ollama (default: model_id)
            quantization: Quantization for GGUF conversion

        Returns:
            Ollama model name
        """
        logger.info(f"Deploying {model_id} to Ollama...")

        if ollama_model_name is None:
            ollama_model_name = model_id

        # Check if already GGUF, else convert
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        if "gguf" not in model.formats:
            logger.info("  Converting to GGUF first...")
            gguf_path = self.convert_to_gguf(model_id, quantization)
        else:
            # Find GGUF file
            mac_locations = dict(self.registry.get_available_locations(model_id))
            model_dir = Path(mac_locations["mac"])
            gguf_files = list(model_dir.glob("*.gguf"))
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {model_dir}")
            gguf_path = gguf_files[0]

        # Create Modelfile
        modelfile = model_dir / "Modelfile"
        modelfile.write_text(f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{ .System }}}}
{{{{ .Prompt }}}}\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
""")

        # Create model in Ollama
        logger.info(f"  Creating Ollama model '{ollama_model_name}'...")

        create_cmd = ["ollama", "create", ollama_model_name, "-f", str(modelfile)]

        try:
            subprocess.run(create_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama create failed: {e.stderr.decode()}")
            raise

        # Update registry
        self.registry.add_backend(model_id, "ollama", ollama_model_name)

        logger.info(f"✓ Deployed to Ollama as '{ollama_model_name}'")
        return ollama_model_name

    def deploy_to_halext(
        self,
        model_id: str,
        node_name: str,
        node_url: str,
    ) -> str:
        """Deploy model to halext AI node.

        Args:
            model_id: Model identifier
            node_name: Node name/ID
            node_url: Node API URL

        Returns:
            Node model ID
        """
        logger.info(f"Deploying {model_id} to halext node '{node_name}'...")

        # Get local model path
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        mac_locations = dict(self.registry.get_available_locations(model_id))
        if "mac" not in mac_locations:
            raise ValueError("Model not available locally - pull first")

        model_path = Path(mac_locations["mac"])

        # TODO: Implement halext node deployment API
        # This would use the halext node API to:
        # 1. Upload model files
        # 2. Register model in node
        # 3. Start serving

        logger.warning("Halext node deployment not yet implemented")

        # Update registry
        self.registry.add_backend(model_id, "halext-node", node_name)

        logger.info(f"✓ Deployed to halext node '{node_name}'")
        return node_name

    def test_model(self, model_id: str, backend: ServingBackend) -> dict:
        """Test deployed model.

        Args:
            model_id: Model identifier
            backend: Backend to test (ollama, llama.cpp, halext-node)

        Returns:
            Test results
        """
        logger.info(f"Testing {model_id} on {backend}...")

        model = self.registry.get_model(model_id)
        if not model or backend not in model.deployed_backends:
            raise ValueError(f"Model {model_id} not deployed to {backend}")

        test_prompt = "Write a simple NOP instruction in 65816 assembly:"

        if backend == "ollama":
            # Test with Ollama
            ollama_name = model.ollama_model_name
            if not ollama_name:
                raise ValueError("Ollama model name not set")

            cmd = ["ollama", "run", ollama_name, test_prompt]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30, check=True
                )
                response = result.stdout.strip()

                logger.info(f"✓ Test passed - Response: {response[:100]}...")

                return {
                    "status": "success",
                    "backend": backend,
                    "prompt": test_prompt,
                    "response": response,
                    "tested_at": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Test failed: {e}")
                return {
                    "status": "failed",
                    "backend": backend,
                    "error": str(e),
                    "tested_at": datetime.now().isoformat(),
                }

        # TODO: Implement tests for other backends

        logger.warning(f"Testing not implemented for {backend}")
        return {"status": "not_implemented", "backend": backend}
