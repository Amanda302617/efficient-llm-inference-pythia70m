python -m src.benchmark_speed `
  --prompt "Language models can be accelerated by" `
  --new-tokens 32 `
  --warmup 1 `
  --runs 3 `
  --output results/baseline_speed.json

