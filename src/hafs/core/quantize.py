"""Quantized embedding operations with NumPy fallback."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

# Try native implementation
_NATIVE_AVAILABLE = False
_HAS_QUANTIZE = False

try:
    from hafs.core._native import (
        quantize_f32_to_int8 as _native_quantize_int8,
        dequantize_int8_to_f32 as _native_dequantize_int8,
        quantize_f32_to_f16 as _native_quantize_f16,
        dequantize_f16_to_f32 as _native_dequantize_f16,
        cosine_similarity_int8 as _native_cosine_int8,
    )
    from hafs.core._native import __has_quantize__

    _NATIVE_AVAILABLE = True
    _HAS_QUANTIZE = __has_quantize__
except ImportError:
    pass


def quantize_to_int8(
    embeddings: NDArray[np.float32],
) -> Tuple[NDArray[np.int8], float, int]:
    """Quantize float32 embeddings to int8.

    Uses symmetric quantization centered at 0. Scale is computed as max_abs/127.

    Args:
        embeddings: Float32 array to quantize

    Returns:
        Tuple of (quantized_data, scale, zero_point)
        - quantized_data: int8 array
        - scale: float scale factor (float_val = int8_val * scale)
        - zero_point: always 0 for symmetric quantization
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if _NATIVE_AVAILABLE and _HAS_QUANTIZE:
        return _native_quantize_int8(embeddings)

    # NumPy fallback: symmetric quantization
    max_abs = np.abs(embeddings).max()
    scale = float(max_abs / 127.0) if max_abs > 1e-8 else 1.0
    quantized = np.round(embeddings / scale).astype(np.int8)
    return quantized, scale, 0


def dequantize_from_int8(
    quantized: NDArray[np.int8],
    scale: float,
    zero_point: int = 0,
) -> NDArray[np.float32]:
    """Dequantize int8 back to float32.

    Args:
        quantized: Int8 quantized array
        scale: Scale factor from quantization
        zero_point: Zero point (usually 0 for symmetric)

    Returns:
        Float32 dequantized array
    """
    quantized = np.ascontiguousarray(quantized, dtype=np.int8)

    if _NATIVE_AVAILABLE and _HAS_QUANTIZE:
        return _native_dequantize_int8(quantized, scale, zero_point)

    return (quantized.astype(np.float32) - zero_point) * scale


def quantize_to_f16(
    embeddings: NDArray[np.float32],
) -> NDArray[np.uint16]:
    """Convert float32 to float16 (stored as uint16).

    This provides 2x memory reduction with minimal precision loss for
    typical embedding values.

    Args:
        embeddings: Float32 array to convert

    Returns:
        Uint16 array containing float16 bit patterns
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if _NATIVE_AVAILABLE and _HAS_QUANTIZE:
        return _native_quantize_f16(embeddings)

    # NumPy fallback using float16
    return embeddings.astype(np.float16).view(np.uint16)


def dequantize_from_f16(
    quantized: NDArray[np.uint16],
) -> NDArray[np.float32]:
    """Convert float16 (uint16) back to float32.

    Args:
        quantized: Uint16 array containing float16 bit patterns

    Returns:
        Float32 array
    """
    quantized = np.ascontiguousarray(quantized, dtype=np.uint16)

    if _NATIVE_AVAILABLE and _HAS_QUANTIZE:
        return _native_dequantize_f16(quantized)

    # NumPy fallback
    return quantized.view(np.float16).astype(np.float32)


def cosine_similarity_int8(
    a: NDArray[np.int8],
    b: NDArray[np.int8],
    scale_a: float,
    scale_b: float,
) -> float:
    """Compute cosine similarity between two int8 quantized vectors.

    Uses SIMD-accelerated int8 dot product when available.

    Args:
        a: First int8 vector
        b: Second int8 vector
        scale_a: Scale factor for vector a
        scale_b: Scale factor for vector b

    Returns:
        Cosine similarity in range [-1, 1]
    """
    a = np.ascontiguousarray(a, dtype=np.int8)
    b = np.ascontiguousarray(b, dtype=np.int8)

    if _NATIVE_AVAILABLE and _HAS_QUANTIZE:
        return _native_cosine_int8(a, b, scale_a, scale_b)

    # NumPy fallback: compute in int32 to avoid overflow
    a_f = a.astype(np.float32) * scale_a
    b_f = b.astype(np.float32) * scale_b
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(dot / (norm_a * norm_b))


def get_quantize_backend_info() -> Dict[str, str]:
    """Get information about the quantization backend.

    Returns:
        Dict with 'native', 'quantize' keys
    """
    return {
        "native": "yes" if _NATIVE_AVAILABLE else "no",
        "quantize": "yes" if _HAS_QUANTIZE else "no",
    }
