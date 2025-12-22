"""Compatibility helpers for Google GenAI SDK variants."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_GENAI_MOD: Any | None = None
_GENAI_IMPORT_ERROR: Exception | None = None
_LEGACY_MOD: Any | None = None
_LEGACY_IMPORT_ERROR: Exception | None = None


def _load_genai_module() -> Any | None:
    """Load google-genai module if available."""
    global _GENAI_MOD, _GENAI_IMPORT_ERROR
    if _GENAI_MOD is not None:
        return _GENAI_MOD
    if _GENAI_IMPORT_ERROR is not None:
        return None
    try:
        import google.genai as genai_mod
    except Exception as exc:
        _GENAI_IMPORT_ERROR = exc
        return None
    _GENAI_MOD = genai_mod
    return _GENAI_MOD


def _load_legacy_module() -> Any | None:
    """Load legacy google-generativeai module if available."""
    global _LEGACY_MOD, _LEGACY_IMPORT_ERROR
    if _LEGACY_MOD is not None:
        return _LEGACY_MOD
    if _LEGACY_IMPORT_ERROR is not None:
        return None
    try:
        import google.generativeai as legacy_mod
    except Exception as exc:
        _LEGACY_IMPORT_ERROR = exc
        return None
    _LEGACY_MOD = legacy_mod
    return _LEGACY_MOD


def genai_available() -> bool:
    """Return True if any compatible GenAI SDK is available."""
    return _load_genai_module() is not None or _load_legacy_module() is not None


@dataclass
class GenAIClient:
    """Wrapper for either google-genai or legacy google-generativeai."""

    mode: str  # "genai" or "legacy"
    client: Any
    async_client: Any = None


def create_genai_client(api_key: str) -> Optional[GenAIClient]:
    """Create a GenAI client wrapper for available SDKs."""
    genai_mod = _load_genai_module()
    if genai_mod is not None:
        try:
            client = genai_mod.Client(api_key=api_key)
            async_client = getattr(client, "aio", None)
            if async_client is None:
                async_cls = getattr(genai_mod, "AsyncClient", None)
                if async_cls:
                    try:
                        async_client = async_cls(api_key=api_key)
                    except Exception:
                        async_client = None
            return GenAIClient(mode="genai", client=client, async_client=async_client)
        except Exception as exc:
            logger.warning("google-genai client init failed: %s", exc)

    legacy_mod = _load_legacy_module()
    if legacy_mod is not None:
        try:
            legacy_mod.configure(api_key=api_key)
            return GenAIClient(mode="legacy", client=legacy_mod)
        except Exception as exc:
            logger.warning("google-generativeai init failed: %s", exc)

    return None


async def generate_content(client: GenAIClient, model: str, contents: str) -> Any:
    """Generate content using a compatible GenAI client."""
    if client.mode == "genai":
        async_models = getattr(client.async_client, "models", None) if client.async_client else None
        if async_models and hasattr(async_models, "generate_content"):
            return await async_models.generate_content(model=model, contents=contents)

        models = getattr(client.client, "models", None)
        if models and hasattr(models, "generate_content"):
            return await asyncio.to_thread(
                models.generate_content,
                model=model,
                contents=contents,
            )

        generate_fn = getattr(client.client, "generate_content", None)
        if generate_fn:
            return await asyncio.to_thread(generate_fn, model=model, contents=contents)

        raise RuntimeError("google-genai client missing generate_content")

    model_client = client.client.GenerativeModel(model)
    return await asyncio.to_thread(model_client.generate_content, contents)


async def embed_content(client: GenAIClient, model: str, contents: str) -> Any:
    """Generate embeddings using a compatible GenAI client."""
    if client.mode == "genai":
        async_models = getattr(client.async_client, "models", None) if client.async_client else None
        if async_models and hasattr(async_models, "embed_content"):
            return await async_models.embed_content(model=model, contents=contents)

        models = getattr(client.client, "models", None)
        if models and hasattr(models, "embed_content"):
            return await asyncio.to_thread(
                models.embed_content,
                model=model,
                contents=contents,
            )

        embed_fn = getattr(client.client, "embed_content", None)
        if embed_fn:
            return await asyncio.to_thread(embed_fn, model=model, contents=contents)

        raise RuntimeError("google-genai client missing embed_content")

    return await asyncio.to_thread(
        client.client.embed_content,
        model=model,
        content=contents,
    )


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_text(response: Any) -> str:
    """Extract response text from GenAI responses."""
    if response is None:
        return ""
    text = _get_field(response, "text")
    if text:
        return text

    candidates = _get_field(response, "candidates")
    if not candidates:
        return ""

    parts_text: list[str] = []
    for candidate in candidates:
        content = _get_field(candidate, "content")
        parts = _get_field(content, "parts") if content else _get_field(candidate, "parts")
        if not parts:
            continue
        for part in parts:
            part_text = _get_field(part, "text")
            if part_text:
                parts_text.append(part_text)

    return "".join(parts_text)


def extract_usage_tokens(response: Any) -> int:
    """Extract token usage from GenAI responses."""
    usage = _get_field(response, "usage_metadata")
    if not usage:
        return 0
    total = _get_field(usage, "total_token_count")
    if isinstance(total, int):
        return total
    prompt = _get_field(usage, "prompt_token_count") or 0
    candidates = _get_field(usage, "candidates_token_count") or 0
    if isinstance(prompt, int) or isinstance(candidates, int):
        return int(prompt) + int(candidates)
    return 0


def extract_embeddings(response: Any) -> list[float]:
    """Extract embedding vector from GenAI embedding responses."""
    if response is None:
        return []

    if isinstance(response, dict):
        embedding = response.get("embedding") or response.get("embeddings")
        if isinstance(embedding, list):
            if embedding and isinstance(embedding[0], (float, int)):
                return [float(v) for v in embedding]
            if embedding and isinstance(embedding[0], dict) and "values" in embedding[0]:
                return [float(v) for v in embedding[0]["values"]]
        if isinstance(embedding, dict) and "values" in embedding:
            return [float(v) for v in embedding["values"]]

    embeddings = _get_field(response, "embeddings")
    if embeddings:
        first = embeddings[0]
        values = _get_field(first, "values")
        if values:
            return [float(v) for v in values]

    return []


def extract_candidate_parts(response: Any) -> tuple[list[str], list[str], list[dict]]:
    """Extract content and thought parts from GenAI responses."""
    content_parts: list[str] = []
    thought_parts: list[str] = []
    raw_parts: list[dict] = []

    candidates = _get_field(response, "candidates")
    if not candidates:
        return content_parts, thought_parts, raw_parts

    for candidate in candidates:
        content = _get_field(candidate, "content")
        parts = _get_field(content, "parts") if content else _get_field(candidate, "parts")
        if not parts:
            continue
        for part in parts:
            part_type = type(part).__name__
            raw_parts.append({"type": part_type, "data": str(part)})

            part_text = _get_field(part, "text")
            if part_text:
                content_parts.append(part_text)

            thought = _get_field(part, "thought") or _get_field(part, "thought_signature")
            if thought:
                thought_parts.append(str(thought))

            pb = getattr(part, "_pb", None)
            pb_thought = getattr(pb, "thought", None) if pb else None
            if pb_thought:
                thought_parts.append(pb_thought)

    return content_parts, thought_parts, raw_parts
