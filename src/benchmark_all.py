"""Run baseline and Streaming KV experiments and summarize results."""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

from .benchmark_speed import DEFAULT_PROMPT, run_speed_benchmark
from .eval_ppl import compute_ppl, compute_streaming_ppl, load_texts
from .utils import (
    DEFAULT_MODEL,
    get_peak_cuda_memory_mb,
    load_model_and_tokenizer,
    reset_cuda_peak_memory,
    save_json,
    set_seed,
)


SUMMARY_FIELDS = [
    "task",
    "dataset",
    "method",
    "model",
    "device",
    "sink_size",
    "recent_size",
    "max_samples",
    "max_length",
    "stride",
    "prompt_tokens",
    "new_tokens",
    "runs",
    "decode_impl",
    "perplexity",
    "total_time_seconds",
    "tokens_per_second",
    "latency_per_token_seconds",
    "peak_cuda_memory_mb",
    "crop_events",
    "max_original_cache_length_seen",
    "max_retained_cache_length_seen",
    "theoretical_compression_ratio",
    "effective_cache_compression_ratio",
    "warning",
    "error",
]


TABLE_COLUMNS = [
    ("Task", "task"),
    ("Dataset", "dataset"),
    ("Method", "method"),
    ("Decode", "decode_impl"),
    ("Sink", "sink_size"),
    ("Recent", "recent_size"),
    ("PPL", "perplexity"),
    ("Tokens/s", "tokens_per_second"),
    ("Latency/token", "latency_per_token_seconds"),
    ("Peak Memory MB", "peak_cuda_memory_mb"),
    ("Max Original", "max_original_cache_length_seen"),
    ("Max Retained", "max_retained_cache_length_seen"),
    ("Compression", "effective_cache_compression_ratio"),
    ("Crop Events", "crop_events"),
    ("Warning", "warning"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all configured benchmark experiments.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc.")
    parser.add_argument("--dtype", default="auto", help="auto, float32, float16, bfloat16.")
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "pg19"])
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--recent-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument(
        "--decode-impl",
        default="manual",
        choices=["generate", "manual"],
        help="Baseline speed decoder. Streaming speed always uses manual.",
    )
    parser.add_argument("--include-pg19", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def base_record(args: argparse.Namespace, task: str, dataset: str, method: str, recent_size: int | None) -> dict:
    """Create a normalized summary record with all expected fields."""
    return {
        "task": task,
        "dataset": dataset,
        "method": method,
        "model": args.model,
        "device": args.device,
        "sink_size": args.sink_size if method == "streaming" else None,
        "recent_size": recent_size if method == "streaming" else None,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "stride": args.stride,
        "prompt_tokens": args.prompt_tokens if task == "speed" else None,
        "new_tokens": args.new_tokens if task == "speed" else None,
        "runs": args.runs if task == "speed" else None,
        "decode_impl": None,
        "perplexity": None,
        "total_time_seconds": None,
        "tokens_per_second": None,
        "latency_per_token_seconds": None,
        "peak_cuda_memory_mb": None,
        "crop_events": None,
        "max_original_cache_length_seen": None,
        "max_retained_cache_length_seen": None,
        "theoretical_compression_ratio": None,
        "effective_cache_compression_ratio": None,
        "warning": None,
        "error": None,
    }


def datasets_to_run(args: argparse.Namespace) -> list[tuple[str, int]]:
    datasets = [(args.dataset, args.max_samples)]
    if args.include_pg19 and args.dataset != "pg19":
        datasets.append(("pg19", min(args.max_samples, 1)))
    return datasets


def run_ppl_record(args, model, tokenizer, device, dataset: str, max_samples: int, method: str, recent_size: int | None):
    """Run one PPL experiment and return a summary record."""
    record = base_record(args, "ppl", dataset, method, recent_size)
    record["max_samples"] = max_samples

    try:
        texts = load_texts(dataset, "test", max_samples)
        reset_cuda_peak_memory(device)
        start_time = time.perf_counter()

        if method == "baseline":
            metrics = compute_ppl(model, tokenizer, device, texts, args.max_length, args.stride)
            warning = None
            theoretical_ratio = 1.0
        else:
            metrics = compute_streaming_ppl(
                model=model,
                tokenizer=tokenizer,
                device=device,
                texts=texts,
                max_length=args.max_length,
                sink_size=args.sink_size,
                recent_size=recent_size,
            )
            warning = metrics.get("warning")
            theoretical_ratio = metrics.get("theoretical_compression_ratio")

        record.update(
            {
                "device": str(device),
                "perplexity": metrics.get("ppl"),
                "total_time_seconds": time.perf_counter() - start_time,
                "peak_cuda_memory_mb": get_peak_cuda_memory_mb(device),
                "crop_events": metrics.get("crop_events"),
                "max_original_cache_length_seen": metrics.get("max_original_cache_length_seen"),
                "max_retained_cache_length_seen": metrics.get("max_retained_cache_length_seen"),
                "theoretical_compression_ratio": theoretical_ratio,
                "effective_cache_compression_ratio": metrics.get("effective_cache_compression_ratio"),
                "warning": warning,
            }
        )
    except Exception as exc:  # noqa: BLE001 - continue with later experiments.
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["warning"] = traceback.format_exc(limit=3)
    return record


def run_speed_record(args, model, tokenizer, device, method: str, recent_size: int | None):
    """Run one speed experiment through benchmark_speed.py's reusable function."""
    record = base_record(args, "speed", args.dataset, method, recent_size)
    speed_args = Namespace(
        model=args.model,
        method=method,
        sink_size=args.sink_size,
        recent_size=recent_size if method == "streaming" else args.sink_size + max(args.recent_sizes),
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        decode_impl=args.decode_impl,
        new_tokens=args.new_tokens,
        warmup=args.warmup,
        runs=args.runs,
        device=args.device,
        dtype=args.dtype,
    )

    try:
        result = run_speed_benchmark(speed_args, model, tokenizer, device)
        record.update(
            {
                "device": str(device),
                "decode_impl": result.get("decode_impl"),
                "total_time_seconds": sum(result.get("elapsed_seconds_per_run", [])),
                "tokens_per_second": result.get("tokens_per_second"),
                "latency_per_token_seconds": result.get("latency_per_token_seconds"),
                "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
                "crop_events": result.get("crop_events"),
                "max_original_cache_length_seen": result.get("max_original_cache_length_seen"),
                "max_retained_cache_length_seen": result.get("max_retained_cache_length_seen"),
                "theoretical_compression_ratio": result.get("theoretical_compression_ratio"),
                "effective_cache_compression_ratio": result.get("effective_cache_compression_ratio"),
                "warning": result.get("warning"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["warning"] = traceback.format_exc(limit=3)
    return record


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).replace("\n", " ")
    return text if len(text) <= 120 else text[:117] + "..."


def write_summary_outputs(records: list[dict], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json({"records": records}, output_path / "summary.json")

    with (output_path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})

    header = "| " + " | ".join(title for title, _ in TABLE_COLUMNS) + " |"
    separator = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    rows = [header, separator]
    for record in records:
        rows.append("| " + " | ".join(format_cell(record.get(field)) for _, field in TABLE_COLUMNS) + " |")
    (output_path / "summary_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    records: list[dict] = []

    try:
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    except Exception as exc:  # noqa: BLE001
        for dataset, max_samples in datasets_to_run(args):
            for method, recent_size in [("baseline", None), *[("streaming", size) for size in args.recent_sizes]]:
                record = base_record(args, "ppl", dataset, method, recent_size)
                record["max_samples"] = max_samples
                record["error"] = f"Model load failed: {type(exc).__name__}: {exc}"
                records.append(record)
        write_summary_outputs(records, args.output_dir)
        print(f"Model load failed. Wrote error summaries to {args.output_dir}.")
        return

    for dataset, max_samples in datasets_to_run(args):
        records.append(run_ppl_record(args, model, tokenizer, device, dataset, max_samples, "baseline", None))
        for recent_size in args.recent_sizes:
            records.append(run_ppl_record(args, model, tokenizer, device, dataset, max_samples, "streaming", recent_size))

    records.append(run_speed_record(args, model, tokenizer, device, "baseline", None))
    for recent_size in args.recent_sizes:
        records.append(run_speed_record(args, model, tokenizer, device, "streaming", recent_size))

    write_summary_outputs(records, args.output_dir)
    print(f"Wrote summary files to {args.output_dir}")


if __name__ == "__main__":
    main()
