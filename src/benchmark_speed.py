"""Generation speed benchmark for Pythia-70M.

The script supports two baseline decoding implementations:

- `generate`: Hugging Face `model.generate`.
- `manual`: the same Python greedy loop used by Streaming KV, but without KV
  cache eviction.

Streaming always uses the manual loop so every step can crop `past_key_values`.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from .streaming_kv import crop_past_key_values, estimate_cache_compression, get_cache_seq_len
from .utils import (
    DEFAULT_MODEL,
    Timer,
    get_peak_cuda_memory_mb,
    load_model_and_tokenizer,
    reset_cuda_peak_memory,
    save_json,
    set_seed,
)


DEFAULT_PROMPT = "Language models can be accelerated by"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generation speed.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    parser.add_argument("--method", default="baseline", choices=["baseline", "streaming"])
    parser.add_argument("--sink-size", type=int, default=4, help="Attention sink tokens kept by streaming KV.")
    parser.add_argument("--recent-size", type=int, default=128, help="Recent tokens kept by streaming KV.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Exact prompt token length for speed tests.")
    parser.add_argument(
        "--decode-impl",
        default="generate",
        choices=["generate", "manual"],
        help="Baseline decoder. Streaming always uses manual decoding.",
    )
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc.")
    parser.add_argument("--dtype", default="auto", help="auto, float32, float16, bfloat16.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/baseline_speed.json")
    return parser.parse_args()


def build_prompt_input_ids(tokenizer, device: torch.device, prompt: str, prompt_tokens: int | None) -> torch.Tensor:
    """Build prompt ids with an exact token length.

    A short prompt is repeated at the token-id level and truncated to
    `prompt_tokens`. This guarantees baseline and streaming use identical input
    token ids and avoids silently benchmarking a prompt too short to trigger KV
    eviction.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if prompt_tokens is None:
        return input_ids
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive.")
    if input_ids.size(1) == 0:
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer produced an empty prompt and has no eos token fallback.")
        input_ids = torch.tensor([[eos]], dtype=torch.long, device=device)

    repeats = (prompt_tokens + input_ids.size(1) - 1) // input_ids.size(1)
    return input_ids.repeat(1, repeats)[:, :prompt_tokens]


def cache_seq_length(past_key_values) -> int:
    """Return the real cache sequence length from current `past_key_values`."""
    seq_len = get_cache_seq_len(past_key_values)
    return 0 if seq_len is None else seq_len


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    new_tokens: int,
    prompt_tokens: int | None = None,
):
    """Generate once with Hugging Face `model.generate`."""
    input_ids = build_prompt_input_ids(tokenizer, device, prompt, prompt_tokens)
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )


@torch.no_grad()
def manual_generate_once(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    new_tokens: int,
    prompt_tokens: int | None = None,
    sink_size: int | None = None,
    recent_size: int | None = None,
    crop_cache: bool = False,
) -> dict:
    """Generate with a manual greedy loop.

    The baseline manual mode sets `crop_cache=False`; streaming sets
    `crop_cache=True`, causing the actual `past_key_values` object to be cropped
    after every autoregressive step.
    """
    input_ids = build_prompt_input_ids(tokenizer, device, prompt, prompt_tokens)
    if new_tokens <= 0:
        return {
            "generated_ids": input_ids,
            "generated_tokens": 0,
            "prompt_tokens": input_ids.size(1),
            "max_original_cache_length_seen": 0,
            "max_retained_cache_length_seen": 0,
            "effective_cache_compression_ratio": 1.0,
            "crop_events": 0,
            "warning": None,
        }

    outputs = model(input_ids, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token]

    past_key_values = outputs.past_key_values
    original_len = cache_seq_length(past_key_values)
    max_original_len = original_len
    if crop_cache:
        if sink_size is None or recent_size is None:
            raise ValueError("sink_size and recent_size are required when crop_cache=True.")
        past_key_values = crop_past_key_values(past_key_values, sink_size=sink_size, recent_size=recent_size)
    retained_len = cache_seq_length(past_key_values)
    max_retained_len = retained_len
    crop_events = int(retained_len < original_len)

    while len(generated_tokens) < new_tokens:
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        past_key_values = outputs.past_key_values
        original_len = cache_seq_length(past_key_values)
        max_original_len = max(max_original_len, original_len)
        if crop_cache:
            past_key_values = crop_past_key_values(past_key_values, sink_size=sink_size, recent_size=recent_size)
        retained_len = cache_seq_length(past_key_values)
        max_retained_len = max(max_retained_len, retained_len)
        if retained_len < original_len:
            crop_events += 1

    generated_ids = torch.cat([input_ids, *generated_tokens], dim=1)
    effective_ratio = max_retained_len / max_original_len if max_original_len > 0 else None
    warning = None
    if crop_cache and prompt_tokens is not None and prompt_tokens + new_tokens <= sink_size + recent_size:
        warning = "prompt_tokens + new_tokens <= sink_size + recent_size, so KV eviction may not trigger."

    return {
        "generated_ids": generated_ids,
        "generated_tokens": len(generated_tokens),
        "prompt_tokens": input_ids.size(1),
        "max_original_cache_length_seen": max_original_len,
        "max_retained_cache_length_seen": max_retained_len,
        "effective_cache_compression_ratio": effective_ratio,
        "crop_events": crop_events,
        "warning": warning,
    }


@torch.no_grad()
def streaming_generate_once(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    new_tokens: int,
    sink_size: int,
    recent_size: int,
    prompt_tokens: int | None = None,
) -> dict:
    """Generate once with manual greedy decoding and Streaming KV eviction."""
    return manual_generate_once(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        new_tokens=new_tokens,
        prompt_tokens=prompt_tokens,
        sink_size=sink_size,
        recent_size=recent_size,
        crop_cache=True,
    )


def run_speed_benchmark(args: argparse.Namespace, model, tokenizer, device: torch.device) -> dict:
    """Run speed benchmark for the requested method and return result data."""
    effective_decode_impl = "manual" if args.method == "streaming" else args.decode_impl

    for _ in range(args.warmup):
        if args.method == "baseline" and args.decode_impl == "generate":
            generate_once(model, tokenizer, device, args.prompt, args.new_tokens, args.prompt_tokens)
        elif args.method == "baseline":
            manual_generate_once(
                model,
                tokenizer,
                device,
                args.prompt,
                args.new_tokens,
                prompt_tokens=args.prompt_tokens,
                crop_cache=False,
            )
        else:
            streaming_generate_once(
                model,
                tokenizer,
                device,
                args.prompt,
                args.new_tokens,
                args.sink_size,
                args.recent_size,
                prompt_tokens=args.prompt_tokens,
            )

    reset_cuda_peak_memory(device)
    elapsed_runs: list[float] = []
    run_details: list[dict] = []

    for _ in range(args.runs):
        with Timer(device) as timer:
            if args.method == "baseline" and args.decode_impl == "generate":
                outputs = generate_once(model, tokenizer, device, args.prompt, args.new_tokens, args.prompt_tokens)
                total_tokens = outputs.size(1)
                details = {
                    "generated_tokens": args.new_tokens,
                    "prompt_tokens": total_tokens - args.new_tokens,
                    "max_original_cache_length_seen": total_tokens,
                    "max_retained_cache_length_seen": total_tokens,
                    "effective_cache_compression_ratio": 1.0,
                    "crop_events": 0,
                    "warning": None,
                }
            elif args.method == "baseline":
                details = manual_generate_once(
                    model,
                    tokenizer,
                    device,
                    args.prompt,
                    args.new_tokens,
                    prompt_tokens=args.prompt_tokens,
                    crop_cache=False,
                )
            else:
                details = streaming_generate_once(
                    model,
                    tokenizer,
                    device,
                    args.prompt,
                    args.new_tokens,
                    args.sink_size,
                    args.recent_size,
                    prompt_tokens=args.prompt_tokens,
                )
        elapsed_runs.append(timer.elapsed)
        run_details.append({key: value for key, value in details.items() if key != "generated_ids"})

    total_generated = args.new_tokens * args.runs
    total_time = sum(elapsed_runs)
    tokens_per_second = total_generated / total_time if total_time > 0 else 0.0
    latency_per_token_ms = (total_time / total_generated) * 1000 if total_generated > 0 else 0.0
    max_total_tokens = max(detail["prompt_tokens"] + detail["generated_tokens"] for detail in run_details)
    if args.method == "streaming":
        compression = estimate_cache_compression(max_total_tokens, args.sink_size, args.recent_size)
    else:
        compression = {
            "retained_tokens": max_total_tokens,
            "original_tokens": max_total_tokens,
            "compression_ratio": 1.0,
        }

    peak_cuda_memory = get_peak_cuda_memory_mb(device)
    max_original = max(detail["max_original_cache_length_seen"] for detail in run_details)
    max_retained = max(detail["max_retained_cache_length_seen"] for detail in run_details)
    effective_ratio = max_retained / max_original if max_original > 0 else None
    warning = next((detail["warning"] for detail in run_details if detail.get("warning")), None)

    return {
        "method": args.method,
        "sink_size": args.sink_size if args.method == "streaming" else None,
        "recent_size": args.recent_size if args.method == "streaming" else None,
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": args.prompt_tokens,
        "decode_impl": effective_decode_impl,
        "new_tokens": args.new_tokens,
        "generated_tokens": total_generated,
        "warmup": args.warmup,
        "runs": args.runs,
        "device": str(device),
        "dtype": args.dtype,
        "elapsed_seconds_per_run": elapsed_runs,
        "mean_elapsed_seconds": statistics.mean(elapsed_runs),
        "tokens_per_second": tokens_per_second,
        "latency_per_token_ms": latency_per_token_ms,
        "latency_per_token_seconds": latency_per_token_ms / 1000,
        "theoretical_cache_tokens": compression["retained_tokens"],
        "theoretical_original_cache_tokens": compression["original_tokens"],
        "theoretical_compression_ratio": compression["compression_ratio"],
        "max_original_cache_length_seen": max_original,
        "max_retained_cache_length_seen": max_retained,
        "effective_cache_compression_ratio": effective_ratio,
        "crop_events": sum(detail["crop_events"] for detail in run_details),
        "warning": warning,
        "run_details": run_details,
        "peak_cuda_memory": peak_cuda_memory,
        "peak_cuda_memory_mb": peak_cuda_memory,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    result = run_speed_benchmark(args, model, tokenizer, device)
    save_json(result, args.output)
    print(f"Saved {args.method} speed results to {args.output}")
    print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
    print(f"Latency/token: {result['latency_per_token_ms']:.2f} ms")


if __name__ == "__main__":
    main()
