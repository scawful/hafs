"""Distributed data generation across multiple nodes.

Architecture:
- Gemini Flash: Primary teacher model (fast, high quality)
- medical-mechanica qwen3:14b: Secondary teacher + quality validation
- medical-mechanica embeddinggemma: Embedding generation
- Load balancing across nodes based on availability

Expected speedup: 2-3x (20-30 hours → 8-12 hours for 34.5K)
"""

import asyncio
import logging
from typing import Callable, Optional

from hafs.backends import OllamaBackend
from hafs.core.orchestrator_v2 import Provider, TaskTier, UnifiedOrchestrator
from agents.training.base import DataGenerator, SourceItem, TrainingSample

logger = logging.getLogger(__name__)

# medical-mechanica config
MEDICAL_MECHANICA_HOST = "100.104.53.21"
MEDICAL_MECHANICA_PORT = 11434

# Model routing
TEACHER_MODELS = {
    "primary": "gemini-3-flash-preview",  # Gemini (fast, good quality)
    "secondary": "qwen3:14b",  # medical-mechanica (coding specialist)
    "reasoning": "deepseek-r1:14b",  # medical-mechanica (complex tasks)
    "validation": "qwen3:14b",  # medical-mechanica (quality validation)
    "embedding": "embeddinggemma",  # medical-mechanica (embeddings)
}


class DistributedGenerationMixin:
    """Mixin to add distributed generation capabilities to DataGenerator."""

    def __init__(self):
        self._mm_ollama: Optional[OllamaBackend] = None
        self._mm_orchestrator: Optional[UnifiedOrchestrator] = None
        self._generation_counter = 0
        self._use_distributed = True

    async def _setup_distributed(self):
        """Initialize medical-mechanica connection."""
        try:
            # Create Ollama backend for medical-mechanica
            self._mm_ollama = OllamaBackend(
                base_url=f"http://{MEDICAL_MECHANICA_HOST}:{MEDICAL_MECHANICA_PORT}"
            )

            # Test connection
            models = await self._mm_ollama.list_models()
            logger.info(f"medical-mechanica available models: {len(models)}")
            self._use_distributed = True
        except Exception as e:
            logger.warning(f"medical-mechanica unavailable: {e}")
            self._use_distributed = False

    async def generate_sample_distributed(
        self, item: SourceItem
    ) -> Optional[TrainingSample]:
        """Generate sample with load balancing across nodes.

        Routes to:
        - Gemini Flash (70%): Fast, good quality
        - medical-mechanica qwen3 (30%): Offload, coding specialist
        """
        self._generation_counter += 1

        # Route to medical-mechanica for 30% of samples
        if self._use_distributed and self._generation_counter % 10 < 3:
            return await self._generate_on_medical_mechanica(item)
        else:
            return await self.generate_sample(item)

    async def _generate_on_medical_mechanica(
        self, item: SourceItem
    ) -> Optional[TrainingSample]:
        """Generate sample on medical-mechanica (qwen3:14b)."""
        if not self._mm_ollama:
            # Fallback to Gemini
            return await self.generate_sample(item)

        try:
            import json
            from agents.training.base import TrainingSample

            prompt = self.get_teacher_prompt(item)

            # Use qwen3:14b on medical-mechanica
            response = await asyncio.wait_for(
                self._mm_ollama.generate(
                    model=TEACHER_MODELS["secondary"],
                    prompt=prompt,
                    temperature=0.7,
                ),
                timeout=45.0,  # Longer timeout for remote
            )

            # Parse response (same logic as generate_sample)
            content = response.get("response", "")

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "{" in content:
                content = content[content.find("{") : content.rfind("}") + 1]

            data = json.loads(content)

            instruction = str(data.get("instruction", "")).strip()
            input_text = str(data.get("input", "")).strip()
            output = str(data.get("output", "")).strip()

            # Get KG entities from item
            kg_entities = []
            if hasattr(item, "name"):
                kg_entities.append(str(item.name))
            if hasattr(item, "memory_access"):
                kg_entities.extend([str(m) for m in item.memory_access])

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain=self.domain,
                source=str(getattr(item, "source", "unknown")),
                teacher_model="qwen3:14b@medical-mechanica",
                teacher_prompt=str(prompt),
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"medical-mechanica timeout for {item.name}, falling back")
            return await self.generate_sample(item)
        except Exception as e:
            logger.warning(f"medical-mechanica error for {item.name}: {e}, falling back")
            return await self.generate_sample(item)


async def generate_embeddings_distributed(
    texts: list[str], use_remote: bool = True
) -> list[list[float]]:
    """Generate embeddings using medical-mechanica embeddinggemma.

    Much faster than Gemini for embeddings.

    Args:
        texts: List of texts to embed
        use_remote: Use medical-mechanica if available

    Returns:
        List of embedding vectors
    """
    if not use_remote:
        # Fallback to orchestrator
        from hafs.core.orchestrator_v2 import UnifiedOrchestrator

        orch = UnifiedOrchestrator()
        embeddings = []
        for text in texts:
            emb = await orch.embed(text)
            embeddings.append(emb)
        return embeddings

    try:
        # Use medical-mechanica embeddinggemma
        mm_ollama = OllamaBackend(
            base_url=f"http://{MEDICAL_MECHANICA_HOST}:{MEDICAL_MECHANICA_PORT}"
        )

        embeddings = []
        for text in texts:
            response = await mm_ollama.generate(
                model=TEACHER_MODELS["embedding"],
                prompt=text,
                options={"embedding_only": True},
            )
            if "embedding" in response:
                embeddings.append(response["embedding"])
            else:
                # Fallback
                from hafs.core.orchestrator_v2 import UnifiedOrchestrator

                orch = UnifiedOrchestrator()
                emb = await orch.embed(text)
                embeddings.append(emb)

        return embeddings

    except Exception as e:
        logger.warning(f"medical-mechanica embedding failed: {e}, using Gemini")
        # Fallback to Gemini
        from hafs.core.orchestrator_v2 import UnifiedOrchestrator

        orch = UnifiedOrchestrator()
        embeddings = []
        for text in texts:
            emb = await orch.embed(text)
            embeddings.append(emb)
        return embeddings


async def validate_quality_distributed(
    sample: TrainingSample, use_remote: bool = True
) -> tuple[bool, dict]:
    """Validate sample quality using medical-mechanica qwen3:14b.

    Offloads quality validation to reduce Gemini load.

    Args:
        sample: Sample to validate
        use_remote: Use medical-mechanica if available

    Returns:
        (is_valid, validation_details)
    """
    if not use_remote:
        # Use local validation
        from agents.training.quality import QualityPipeline

        pipeline = QualityPipeline()
        return await pipeline.validate(sample)

    try:
        mm_ollama = OllamaBackend(
            base_url=f"http://{MEDICAL_MECHANICA_HOST}:{MEDICAL_MECHANICA_PORT}"
        )

        # Use qwen3 for validation
        prompt = f"""Validate this training sample for quality:

INSTRUCTION: {sample.instruction[:300]}
OUTPUT: {sample.output[:500]}

Check:
1. Is the output relevant to the instruction?
2. Is the output technically correct (for code domains)?
3. Is there any hallucinated/incorrect information?

Respond with JSON:
{{"valid": true/false, "score": 0.0-1.0, "reason": "explanation"}}"""

        response = await asyncio.wait_for(
            mm_ollama.generate(
                model=TEACHER_MODELS["validation"],
                prompt=prompt,
                temperature=0.3,
            ),
            timeout=120.0,  # Increased for GPU/slower models
        )

        import json

        content = response.get("response", "")
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "{" in content:
            content = content[content.find("{") : content.rfind("}") + 1]

        data = json.loads(content)
        is_valid = data.get("valid", True)
        score = data.get("score", 0.5)

        return is_valid, {"valid": is_valid, "score": score, "details": data}

    except Exception as e:
        logger.warning(f"medical-mechanica validation failed: {e}, using local")
        # Fallback to local
        from agents.training.quality import QualityPipeline

        pipeline = QualityPipeline()
        return await pipeline.validate(sample)


class LoadBalancer:
    """Load balancer for distributed generation."""

    def __init__(self):
        self.stats = {
            "gemini": {"requests": 0, "errors": 0, "total_time": 0.0},
            "medical-mechanica": {"requests": 0, "errors": 0, "total_time": 0.0},
        }

    def get_next_provider(self) -> str:
        """Get next provider based on load and health.

        Returns:
            "gemini" or "medical-mechanica"
        """
        gemini_load = self.stats["gemini"]["requests"]
        mm_load = self.stats["medical-mechanica"]["requests"]

        # 70/30 split favoring Gemini (faster, more reliable)
        if mm_load < gemini_load * 0.3:
            return "medical-mechanica"
        else:
            return "gemini"

    def record_request(self, provider: str, duration: float, error: bool = False):
        """Record request stats."""
        if provider in self.stats:
            self.stats[provider]["requests"] += 1
            self.stats[provider]["total_time"] += duration
            if error:
                self.stats[provider]["errors"] += 1

    def get_stats(self) -> dict:
        """Get load balancing stats."""
        return {
            provider: {
                "requests": stats["requests"],
                "errors": stats["errors"],
                "avg_time": (
                    stats["total_time"] / stats["requests"]
                    if stats["requests"] > 0
                    else 0
                ),
                "error_rate": (
                    stats["errors"] / stats["requests"]
                    if stats["requests"] > 0
                    else 0
                ),
            }
            for provider, stats in self.stats.items()
        }
