"""StreamingLLM-style KV cache eviction.

This module implements a training-free inference optimization inspired by
StreamingLLM. It keeps:

1. Keep the first `sink_size` KV cache positions as attention sink tokens.
2. Keep the most recent `recent_size` KV cache positions.
3. Drop the middle KV cache positions during decoding.

The code never changes model weights. It only changes the `past_key_values`
object passed between autoregressive forward calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StreamingKVConfig:
    """Configuration for the sink-plus-recent KV eviction method."""

    sink_size: int = 4
    recent_size: int = 128


def _validate_sizes(sink_size: int, recent_size: int) -> None:
    if sink_size < 0:
        raise ValueError("sink_size must be non-negative.")
    if recent_size < 0:
        raise ValueError("recent_size must be non-negative.")
    if sink_size + recent_size <= 0:
        raise ValueError("sink_size + recent_size must be positive.")


def _sequence_dim(tensor: torch.Tensor) -> int:
    """Return the sequence-length dimension for common HF KV tensors.

    Pythia/GPT-NeoX returns key/value tensors with shape
    `[batch, num_heads, seq_len, head_dim]`, so the sequence length is `-2`.
    Many other Hugging Face causal LMs use the same layout. Some caches may
    store `[batch, seq_len, hidden]`, where `-2` is still the sequence length.

    This project intentionally supports these common tensor layouts. If a model
    uses an unusual KV layout, raising a clear error is safer than silently
    cropping the wrong dimension.
    """
    if tensor.ndim < 3:
        raise ValueError(f"Unsupported KV tensor shape {tuple(tensor.shape)}; expected at least 3 dims.")
    return -2


def _crop_tensor_by_sequence(tensor: torch.Tensor, sink_size: int, recent_size: int) -> torch.Tensor:
    """Crop one key or value tensor along its sequence-length dimension."""
    seq_dim = _sequence_dim(tensor)
    seq_len = tensor.shape[seq_dim]
    keep = sink_size + recent_size
    if seq_len <= keep:
        return tensor

    parts: list[torch.Tensor] = []
    if sink_size > 0:
        parts.append(tensor.narrow(seq_dim, 0, sink_size))
    if recent_size > 0:
        parts.append(tensor.narrow(seq_dim, seq_len - recent_size, recent_size))
    return torch.cat(parts, dim=seq_dim)


def _crop_legacy_cache(
    legacy_cache: tuple[tuple[torch.Tensor, torch.Tensor, Any], ...] | list,
    sink_size: int,
    recent_size: int,
):
    """Crop a legacy Hugging Face cache represented as tuples per layer.

    Legacy causal-LM caches are usually:
    `((key_layer_0, value_layer_0), (key_layer_1, value_layer_1), ...)`.
    Some models include extra per-layer fields, so this function preserves any
    tuple entries after key/value unchanged.
    """
    cropped_layers = []
    for layer_idx, layer_cache in enumerate(legacy_cache):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
            raise ValueError(f"Unsupported cache layer at index {layer_idx}: {type(layer_cache)!r}")
        key, value, *extras = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Cache layer {layer_idx} does not start with key/value tensors.")

        cropped_key = _crop_tensor_by_sequence(key, sink_size, recent_size)
        cropped_value = _crop_tensor_by_sequence(value, sink_size, recent_size)
        cropped_layers.append((cropped_key, cropped_value, *extras))

    return tuple(cropped_layers)


def crop_past_key_values(past_key_values, sink_size: int, recent_size: int):
    """Crop Hugging Face `past_key_values` using sink plus recent retention.

    Supported formats:

    - Legacy tuple/list cache: each layer begins with `(key, value)`.
    - `DynamicCache` and similar Cache objects that expose
      `to_legacy_cache()` plus class method `from_legacy_cache()`.

    The function returns a new cache object in the same broad format. It does
    not mutate model parameters. For DynamicCache, this implementation converts
    to a legacy tuple, crops tensors, then reconstructs a DynamicCache.
    """
    _validate_sizes(sink_size, recent_size)
    if past_key_values is None:
        return None

    if isinstance(past_key_values, (tuple, list)):
        return _crop_legacy_cache(past_key_values, sink_size, recent_size)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        cropped_legacy = _crop_legacy_cache(legacy_cache, sink_size, recent_size)
        cache_class = past_key_values.__class__
        if hasattr(cache_class, "from_legacy_cache"):
            return cache_class.from_legacy_cache(cropped_legacy)

    raise TypeError(
        "Unsupported past_key_values format. Expected a legacy tuple/list cache "
        "or a Cache object with to_legacy_cache/from_legacy_cache support."
    )


def get_cache_seq_len(past_key_values) -> int | None:
    """Return the real sequence length currently stored in `past_key_values`.

    The value is read from the actual cache object/tensors, not estimated from
    benchmark arguments. Returns None if the cache is absent. Raises a clear
    error for unknown cache formats so callers can record an experiment warning
    instead of fabricating statistics.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())

    legacy = past_key_values.to_legacy_cache() if hasattr(past_key_values, "to_legacy_cache") else past_key_values
    if isinstance(legacy, (tuple, list)) and legacy:
        layer = legacy[0]
        if isinstance(layer, (tuple, list)) and layer and isinstance(layer[0], torch.Tensor):
            return int(layer[0].shape[_sequence_dim(layer[0])])

    raise TypeError(
        "Unsupported past_key_values format while reading cache length. "
        "Expected DynamicCache-like object or legacy tuple/list cache."
    )


def estimate_cache_compression(original_length: int, sink_size: int, recent_size: int) -> dict[str, float | int]:
    """Estimate the theoretical token retention ratio for KV cache cropping.

    `compression_ratio` is defined as `retained_tokens / original_tokens`.
    Smaller values mean a smaller KV cache. If the sequence is shorter than the
    sink-plus-recent budget, no cache entries are dropped and the ratio is 1.0.
    """
    _validate_sizes(sink_size, recent_size)
    if original_length < 0:
        raise ValueError("original_length must be non-negative.")
    if original_length == 0:
        return {
            "retained_tokens": 0,
            "original_tokens": 0,
            "compression_ratio": 1.0,
        }

    retained_tokens = min(original_length, sink_size + recent_size)
    return {
        "retained_tokens": retained_tokens,
        "original_tokens": original_length,
        "compression_ratio": retained_tokens / original_length,
    }


def compress_past_key_values(past_key_values, config: StreamingKVConfig):
    """Compatibility wrapper around `crop_past_key_values`."""
    return crop_past_key_values(
        past_key_values=past_key_values,
        sink_size=config.sink_size,
        recent_size=config.recent_size,
    )
