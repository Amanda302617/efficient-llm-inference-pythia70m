| Task | Dataset | Method | Decode | Sink | Recent | PPL | Tokens/s | Latency/token | Peak Memory MB | Max Original | Max Retained | Compression | Crop Events | Warning |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ppl | wikitext2 | baseline | - | - | - | 65.9475 | - | - | - | 196 | 196 | 1.0000 | 0 | - |
| ppl | wikitext2 | streaming | - | 4 | 32 | 263.3388 | - | - | - | 37 | 36 | 0.9730 | 306 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses s... |
| ppl | wikitext2 | streaming | - | 4 | 64 | 210.4257 | - | - | - | 69 | 68 | 0.9855 | 242 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses s... |
| ppl | wikitext2 | streaming | - | 4 | 128 | 96.5670 | - | - | - | 133 | 132 | 0.9925 | 114 | Streaming PPL uses incremental next-token scoring over each sample truncated to max_length, while baseline PPL uses s... |
| speed | wikitext2 | baseline | manual | - | - | - | 35.4861 | 0.0282 | - | 575 | 575 | 1.0000 | 0 | - |
| speed | wikitext2 | streaming | manual | 4 | 32 | - | 35.7622 | 0.0280 | - | 512 | 36 | 0.0703 | 128 | - |
| speed | wikitext2 | streaming | manual | 4 | 64 | - | 35.8526 | 0.0279 | - | 512 | 68 | 0.1328 | 128 | - |
| speed | wikitext2 | streaming | manual | 4 | 128 | - | 34.1813 | 0.0293 | - | 512 | 132 | 0.2578 | 128 | - |
