"""Baseline perplexity evaluation for Pythia-70M.

This script computes causal language-model perplexity without training or
changing model weights. It uses a sliding-window evaluation: each window is fed
to the model with labels equal to input ids, then tokens that overlap with the
previous window are masked out with label value -100.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from .streaming_kv import crop_past_key_values, estimate_cache_compression, get_cache_seq_len
from .utils import (
    DEFAULT_MODEL,
    get_peak_cuda_memory_mb,
    load_model_and_tokenizer,
    reset_cuda_peak_memory,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute baseline perplexity.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    parser.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "streaming"],
        help="Evaluation method: full-cache baseline or Streaming KV.",
    )
    parser.add_argument("--sink-size", type=int, default=4, help="Attention sink tokens kept by streaming KV.")
    parser.add_argument("--recent-size", type=int, default=128, help="Recent tokens kept by streaming KV.")
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        choices=["wikitext2", "pg19"],
        help="Dataset preset to evaluate.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc.")
    parser.add_argument("--dtype", default="auto", help="auto, float32, float16, bfloat16.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/baseline_ppl.json")
    return parser.parse_args()


def load_texts(dataset_name: str, split: str, max_samples: int) -> list[str]:
    """Load a small text subset for evaluation.

    WikiText-2 is the default because it downloads quickly and is common for
    perplexity checks. PG-19 is included for the later assignment experiment,
    but small subsets should be used for fast local iteration.
    """
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_column = "text"
    elif dataset_name == "pg19":
        dataset = load_dataset("pg19", split=split)
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    texts: list[str] = []
    for row in dataset:
        text = row[text_column].strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def iter_windows(input_ids: torch.Tensor, max_length: int, stride: int) -> Iterable[tuple[torch.Tensor, int]]:
    """Yield token windows and the number of new target tokens in each window.

    The window may overlap with the previous one when `stride < max_length`.
    Only tokens after the previous window end are counted as fresh targets.
    Earlier overlapping tokens are used as context and masked out in the loss.
    """
    seq_len = input_ids.size(1)
    previous_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        window_begin = max(0, end - max_length)
        target_len = end - previous_end
        window = input_ids[:, window_begin:end]
        yield window, target_len
        previous_end = end
        if end == seq_len:
            break


@torch.no_grad()
def compute_ppl(model, tokenizer, device: torch.device, texts: list[str], max_length: int, stride: int) -> dict:
    """Compute token-level negative log-likelihood and perplexity.

    For each sliding window, labels are copied from input ids. Tokens that are
    context-only are replaced with -100 so that Transformers excludes them from
    the loss. Perplexity is exp(total negative log-likelihood / target tokens).
    """
    total_nll = 0.0
    total_tokens = 0
    max_window_length_seen = 0

    for text in tqdm(texts, desc="Evaluating PPL"):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        if input_ids.size(1) < 2:
            continue

        for window, target_len in iter_windows(input_ids, max_length=max_length, stride=stride):
            max_window_length_seen = max(max_window_length_seen, window.size(1))
            labels = window.clone()
            context_tokens = labels.size(1) - target_len
            if context_tokens > 0:
                labels[:, :context_tokens] = -100
            if labels[:, 1:].ne(-100).sum().item() == 0:
                continue

            outputs = model(window, labels=labels)
            valid_tokens = labels[:, 1:].ne(-100).sum().item()
            total_nll += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise RuntimeError("No valid target tokens were evaluated. Increase max samples or text length.")

    mean_nll = total_nll / total_tokens
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "evaluated_tokens": total_tokens,
        "max_original_cache_length_seen": max_window_length_seen,
        "max_retained_cache_length_seen": max_window_length_seen,
        "effective_cache_compression_ratio": 1.0,
        "crop_events": 0,
    }


def cache_seq_length(past_key_values) -> int:
    """Read the cache sequence length from either DynamicCache or legacy tuple.

    For Pythia/GPT-NeoX, legacy key tensors have shape
    `[batch, num_heads, seq_len, head_dim]`, so `shape[-2]` is the current
    number of cached token positions.
    """
    seq_len = get_cache_seq_len(past_key_values)
    return 0 if seq_len is None else seq_len


@torch.no_grad()
def compute_streaming_ppl(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    max_length: int,
    sink_size: int,
    recent_size: int,
) -> dict:
    """Compute PPL with incremental Streaming KV next-token scoring.

    This intentionally differs from the baseline sliding-window implementation.
    Each sample is truncated to `max_length` tokens, then scored one token at a
    time. After every forward pass, `past_key_values` is cropped to keep only
    sink tokens plus the recent window. This is slower than a batched baseline
    but directly exercises the KV eviction path used during generation.
    """
    total_nll = 0.0
    total_tokens = 0
    max_original_cache_length = 0
    max_retained_cache_length = 0
    max_theoretical_uncropped_cache_length = 0
    crop_events = 0

    for text in tqdm(texts, desc="Evaluating Streaming PPL"):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded.input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue
        max_theoretical_uncropped_cache_length = max(max_theoretical_uncropped_cache_length, seq_len - 1)

        past_key_values = None
        current_token = input_ids[:, :1]

        for position in range(seq_len - 1):
            outputs = model(current_token, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, position + 1]
            loss = F.cross_entropy(logits.float(), target, reduction="sum")
            total_nll += loss.item()
            total_tokens += 1

            past_key_values = outputs.past_key_values
            original_cache_length = cache_seq_length(past_key_values)
            max_original_cache_length = max(max_original_cache_length, original_cache_length)

            cropped = crop_past_key_values(past_key_values, sink_size=sink_size, recent_size=recent_size)
            retained_cache_length = cache_seq_length(cropped)
            max_retained_cache_length = max(max_retained_cache_length, retained_cache_length)
            if retained_cache_length < original_cache_length:
                crop_events += 1
            past_key_values = cropped

            current_token = input_ids[:, position + 1 : position + 2]

    if total_tokens == 0:
        raise RuntimeError("No valid target tokens were evaluated. Increase max samples or text length.")

    mean_nll = total_nll / total_tokens
    compression = estimate_cache_compression(max_theoretical_uncropped_cache_length, sink_size, recent_size)
    effective_ratio = (
        max_retained_cache_length / max_original_cache_length if max_original_cache_length > 0 else None
    )
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "evaluated_tokens": total_tokens,
        "theoretical_cache_tokens": compression["retained_tokens"],
        "theoretical_original_cache_tokens": compression["original_tokens"],
        "theoretical_compression_ratio": compression["compression_ratio"],
        "max_original_cache_length": max_original_cache_length,
        "max_retained_cache_length": max_retained_cache_length,
        "max_original_cache_length_seen": max_original_cache_length,
        "max_retained_cache_length_seen": max_retained_cache_length,
        "max_theoretical_uncropped_cache_length": max_theoretical_uncropped_cache_length,
        "crop_events": crop_events,
        "effective_cache_compression_ratio": effective_ratio,
        "warning": (
            "Streaming PPL uses incremental next-token scoring over each sample "
            "truncated to max_length, while baseline PPL uses sliding-window "
            "teacher-forced loss. Values are useful for comparing trends but "
            "are not computed by an identical code path."
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    texts = load_texts(args.dataset, args.split, args.max_samples)

    reset_cuda_peak_memory(device)
    if args.method == "baseline":
        metrics = compute_ppl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            texts=texts,
            max_length=args.max_length,
            stride=args.stride,
        )
        compression = estimate_cache_compression(args.max_length, args.sink_size, args.recent_size)
        metrics.update(
            {
                "theoretical_cache_tokens": compression["original_tokens"],
                "theoretical_original_cache_tokens": compression["original_tokens"],
                "theoretical_compression_ratio": 1.0,
                "effective_cache_compression_ratio": 1.0,
                "warning": None,
            }
        )
    else:
        metrics = compute_streaming_ppl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            texts=texts,
            max_length=args.max_length,
            sink_size=args.sink_size,
            recent_size=args.recent_size,
        )

    peak_cuda_memory = get_peak_cuda_memory_mb(device)
    result = {
        "method": args.method,
        "sink_size": args.sink_size if args.method == "streaming" else None,
        "recent_size": args.recent_size if args.method == "streaming" else None,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "loaded_samples": len(texts),
        "max_length": args.max_length,
        "stride": args.stride,
        "device": str(device),
        "dtype": args.dtype,
        "peak_cuda_memory": peak_cuda_memory,
        "peak_cuda_memory_mb": peak_cuda_memory,
        **metrics,
    }
    save_json(result, args.output)
    print(f"Saved {args.method} PPL results to {args.output}")
    print(f"PPL: {result['ppl']:.4f} over {result['evaluated_tokens']} tokens")


if __name__ == "__main__":
    main()
