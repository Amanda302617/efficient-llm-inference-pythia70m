"""Shared helpers for model loading, result saving, and measurement.

The project is intentionally small and explicit so that the course artifact is
easy to inspect and reproduce. No helper in this file trains the model or
changes model parameters.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "EleutherAI/pythia-70m"


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str | None = None) -> torch.device:
    """Return the requested device, or CUDA when available.

    Use `--device cpu` for a small smoke test on machines without a GPU. The
    scripts keep the device choice explicit in their JSON outputs.
    """
    if device is not None and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    dtype: str = "auto",
):
    """Load the tokenizer and causal LM for inference only.

    The model is moved to the selected device and switched to eval mode. All
    parameters are frozen to make it clear that the baseline is inference-only.
    """
    selected_device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    elif dtype != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(selected_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, tokenizer, selected_device


def reset_cuda_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak memory counters when CUDA measurement is available."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_cuda_memory_mb(device: torch.device) -> float | None:
    """Return peak CUDA memory in MiB, or None for CPU runs."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before reading wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """Tiny wall-clock timer with CUDA synchronization support."""

    def __init__(self, device: torch.device):
        self.device = device
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        sync_if_cuda(self.device)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sync_if_cuda(self.device)
        self.elapsed = time.perf_counter() - self.start_time


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save experiment metadata and results as pretty JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

