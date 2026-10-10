# kvcalc

KV cache VRAM calculator and live monitor for LLM serving: predict memory per model and config, then track real block utilisation on vLLM.

Project 3 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

A calculator that answers, before a GPU is rented, "how many tokens of context can this GPU hold for this model, and so how many concurrent requests?", and a monitor that shows live KV cache usage next to that prediction so the calculator can be calibrated against reality. [CONCEPTS.md](CONCEPTS.md) has the math.

## Why it matters

Most production OOMs are KV cache math errors, not model size.

## Usage

```bash
pip install "git+https://github.com/NomanJafar/kvcalc"

# predict from a local config.json, or fetch it from the Hub
kvcalc predict --hf Qwen/Qwen2.5-1.5B-Instruct --vram 15 --max-model-len 4096
kvcalc predict --config ./config.json --params 8.03e9 --vram 24 --utilization 0.9 --shared-prefix 1500 --json

# compare well-known models across GPU sizes
kvcalc table --gpus 16,24,80

# watch a running vLLM server against the prediction
kvcalc monitor --base-url http://localhost:8000 --kv-tokens 360692 --interval 1 --out results/timeline.json
```

`kvcalc predict` output for Qwen2.5-1.5B-Instruct on a T4 (15 GiB usable, utilisation 0.9, FP16):

```
GPU budget: 15.0 GiB x 0.9 = 13.50 GiB
  weights      2.87 GiB
  activations  0.60 GiB (assumed)
  overhead     0.40 GiB (assumed)
  KV cache     9.63 GiB
KV cache: 28.0 KB/token -> 360,692 tokens (22,543 blocks of 16)
max concurrent requests at 4096 tokens each: 88.1
```

The "assumed" lines are the fuzzy part of the model; the GPU validation below measures how far off they are.

## Comparison table (FP16, 4k-token requests, utilisation 0.9)

| model | attn | KV heads | KB/token | 16 GB: 4k-ctx concurrency | 24 GB: 4k-ctx concurrency | 80 GB: 4k-ctx concurrency |
|---|---|---|---|---|---|---|
| meta-llama/Llama-2-7b-hf | MHA | 32 | 512 | 0.4 | 4.0 | 29 |
| meta-llama/Llama-3.1-8B | GQA | 8 | 128 | does not fit | 11 | 112 |
| mistralai/Mistral-7B-v0.3 | GQA | 8 | 128 | does not fit | 14 | 115 |
| Qwen/Qwen2.5-7B-Instruct | GQA | 4 | 56 | does not fit | 29 | 260 |
| Qwen/Qwen2.5-1.5B-Instruct | GQA | 2 | 28 | 96 | 162 | 623 |
| deepseek-ai/DeepSeek-V2-Lite | MLA | 16 | 30 | does not fit | does not fit | 352 |
Three things to notice. Llama-2-7B (MHA, 512 KB/token) "fits" on a 16 GB card with room for 0.4 requests: it loads and falls over under load, which is the failure mode this project exists to prevent. GQA with 2 KV heads makes Qwen2.5-1.5B 18x cheaper per token than Llama-2-7B. DeepSeek-V2-Lite is a 16B-parameter model with a 30 KB/token cache thanks to MLA; the weights are the problem there, not the cache.

Field values in the table come from each model's `config.json` and are embedded in `kvcalc.table.KNOWN_MODELS`; `kvcalc predict --hf` fetches live configs.

## Monitor

`kvcalc monitor` polls `/metrics` and prints, each interval: KV cache usage (and the implied tokens in use against the predicted capacity), running and waiting requests, cumulative preemptions, and generation tokens/s derived from the counter. Waiting > 0 while usage < 100% means the limit is `--max-num-seqs`, not memory; preemptions climbing means the cache is undersized for the load. A JSON timeline is written with `--out`.

## GPU validation

`notebooks/kaggle_kvcalc.ipynb` predicts the KV cache size for Qwen2.5-1.5B at `max-model-len` 2048 and 8192, launches vLLM at each on a T4, and compares the prediction with the server's reported "GPU KV cache size" and "Available KV cache memory"; then it runs the monitor during a `tokenbench` closed-loop sweep to see what caps concurrency. Results will be added here after that run.

## Layout

```
CONCEPTS.md           memory accounting, bytes per token for MHA/GQA/MLA, concurrency math, why servers OOM
src/kvcalc/
  calc.py             ModelSpec.from_config, kv_bytes_per_token, capacity
  table.py            KNOWN_MODELS and the comparison table
  monitor.py          /metrics polling and timeline
  cli.py              predict | table | monitor
tests/                22 tests; the monitor is exercised against tokenbench's mock server
```

## Running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). The activation and overhead defaults are calibrated for that card and vLLM 0.31; other GPUs will need the two assumed terms re-measured.

## License

MIT
