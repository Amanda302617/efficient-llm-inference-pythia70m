# Efficient Inference for Pythia-70M

Repository: https://github.com/Amanda302617/efficient-llm-inference-pythia70m

This repository is a course project for training-free language model inference optimization on `EleutherAI/pythia-70m`.

The implemented method is a StreamingLLM-style KV cache eviction policy. It keeps attention sink tokens plus a recent window, and drops the middle historical KV cache during autoregressive decoding. The model is not trained and its parameters are not modified.

## Motivation

Autoregressive Transformer inference stores previous key/value states in a KV cache. As the context length grows, this cache grows linearly with the number of tokens, increasing memory use and the amount of attention computation during decoding.

This project studies a training-free way to limit that KV cache budget. Instead of fine-tuning the model or changing model weights, it compresses the cache at inference time by keeping a small set of important historical positions and discarding the middle history.

## Method

During decoding, standard Transformer inference stores previous key/value tensors in `past_key_values`. The cache grows with sequence length. This project implements a simple per-layer KV cache compression policy:

- Baseline: keep the full KV cache.
- Streaming KV: crop `past_key_values` after each autoregressive step.
- Keep the first `sink_size` tokens as attention sink tokens.
- Keep the most recent `recent_size` tokens.
- Drop intermediate historical KV states.
- Apply the policy to both legacy Hugging Face tuple caches and `DynamicCache`.

For Pythia/GPT-NeoX style tensors, key/value cache tensors have shape `[batch, num_heads, seq_len, head_dim]`, so the sequence length dimension is `-2`.

## Installation

```bash
pip install -r requirements.txt
```

If you use CUDA, install a PyTorch build that matches your CUDA version.

## Run Benchmarks

Small CPU smoke benchmark:

```bash
python -m src.benchmark_all --device cpu --max-samples 2 --max-length 128 --stride 128 --prompt-tokens 128 --new-tokens 8 --runs 1 --warmup 0 --recent-sizes 32 64 --sink-size 4 --decode-impl manual --output-dir results
```

Effective WikiText-2 benchmark used for the current report:

```bash
python -m src.benchmark_all --device cpu --max-samples 4 --max-length 512 --stride 512 --prompt-tokens 512 --new-tokens 64 --runs 2 --warmup 1 --recent-sizes 32 64 128 --sink-size 4 --decode-impl manual --output-dir results
```

Optional PG-19 subset benchmark:

```bash
python -m src.benchmark_all --device cpu --include-pg19 --max-samples 1 --max-length 512 --stride 512 --prompt-tokens 512 --new-tokens 32 --runs 1 --warmup 0 --recent-sizes 128 --sink-size 4 --decode-impl manual --output-dir results
```

The benchmark writes:

- `results/summary.json`
- `results/summary.csv`
- `results/summary_table.md`

## Experiment Setup

The reported WikiText-2 benchmark uses:

- Model: `EleutherAI/pythia-70m`
- Dataset: WikiText-2 test split
- Device: CPU
- `max_samples=4`
- `max_length=512`
- `stride=512`
- `prompt_tokens=512`
- `new_tokens=64`
- `runs=2`
- `warmup=1`
- `sink_size=4`
- `recent_sizes=32,64,128`
- `decode_impl=manual`

## Results

The following table comes from an actual WikiText-2 CPU run, not from the earlier smoke test.

Command:

```bash
python -m src.benchmark_all --device cpu --max-samples 4 --max-length 512 --stride 512 --prompt-tokens 512 --new-tokens 64 --runs 2 --warmup 1 --recent-sizes 32 64 128 --sink-size 4 --decode-impl manual --output-dir results
```

| Task | Dataset | Method | Decode | Sink | Recent | PPL | Tokens/s | Latency/token | Peak Memory MB | Max Original | Max Retained | Compression | Crop Events | Warning |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ppl | wikitext2 | baseline | - | - | - | 65.9475 | - | - | - | 196 | 196 | 1.0000 | 0 | - |
| ppl | wikitext2 | streaming | - | 4 | 32 | 263.3388 | - | - | - | 37 | 36 | 0.9730 | 306 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses sliding-window teacher-forced loss. |
| ppl | wikitext2 | streaming | - | 4 | 64 | 210.4257 | - | - | - | 69 | 68 | 0.9855 | 242 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses sliding-window teacher-forced loss. |
| ppl | wikitext2 | streaming | - | 4 | 128 | 96.5670 | - | - | - | 133 | 132 | 0.9925 | 114 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses sliding-window teacher-forced loss. |
| speed | wikitext2 | baseline | manual | - | - | - | 35.4861 | 0.0282 | - | 575 | 575 | 1.0000 | 0 | - |
| speed | wikitext2 | streaming | manual | 4 | 32 | - | 35.7622 | 0.0280 | - | 512 | 36 | 0.0703 | 128 | - |
| speed | wikitext2 | streaming | manual | 4 | 64 | - | 35.8526 | 0.0279 | - | 512 | 68 | 0.1328 | 128 | - |
| speed | wikitext2 | streaming | manual | 4 | 128 | - | 34.1813 | 0.0293 | - | 512 | 132 | 0.2578 | 128 | - |

## Discussion

The Streaming KV implementation is active in the speed benchmark. For all streaming speed runs, `crop_events=128`, `max_retained_cache_length_seen < max_original_cache_length_seen`, and `effective_cache_compression_ratio < 1`. This shows that KV cache eviction happened in the actual `past_key_values`, not only in a theoretical estimate.

With `recent_size=32`, the speed benchmark retained at most 36 cache positions out of an original 512, giving an effective cache compression ratio of 0.0703. With `recent_size=64`, the ratio increased to 0.1328. With `recent_size=128`, the ratio increased to 0.2578. As expected, larger recent windows preserve more context but compress the cache less.

The PPL trend also follows the expected direction: smaller recent windows lose more historical context and produce worse perplexity. Streaming PPL is much worse with `recent_size=32` than with `recent_size=128`.

Wall-clock speed on CPU is mixed. `recent_size=32` and `recent_size=64` are slightly faster than the manual baseline in this run, while `recent_size=128` is slower. This should not be overclaimed. On CPU with a small Pythia-70M model, Python-level greedy decoding, `DynamicCache` conversion, and cache slicing overhead can offset theoretical attention-cache savings.

## Limitations

- The current environment uses CPU PyTorch, so CUDA memory results are unavailable.
- Pythia-70M is small, so KV cache is not the only performance bottleneck.
- Baseline PPL and streaming PPL use different computation paths. Baseline PPL uses sliding-window teacher-forced loss, while streaming PPL uses incremental next-token scoring with cache eviction. PPL values should be read as a trend comparison, not a perfectly controlled one-to-one metric.
- The benchmark uses a small number of WikiText-2 samples and is not intended to claim SOTA performance.
- The manual Python decoding loop is useful for demonstrating the algorithm, but it is not equivalent to an optimized production inference kernel.

## Reproducibility

Environment used for the reported run:

- Python: 3.10.18
- PyTorch: 2.11.0+cpu
- Transformers: 4.57.3
- Datasets: 4.4.2
- CUDA available: false
- Model: `EleutherAI/pythia-70m`
- Dataset: WikiText-2 (`wikitext`, `wikitext-2-raw-v1`, test split)

The reported numbers are generated by `src.benchmark_all` and saved in `results/summary.csv`; they are not manually invented. The repository does not include model weights, Hugging Face dataset caches, or local cache directories. Install dependencies from `requirements.txt`, then run the benchmark command above to regenerate the result files.
