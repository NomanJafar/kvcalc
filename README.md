# kvcalc

KV cache VRAM calculator and live monitor for LLM serving: predict memory per model and config, then track real block utilisation on vLLM.

Project 3 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Predict KV cache VRAM for a given model, dtype, context length and batch, then track live cache utilisation on a running server.

## Why it matters

Most production OOMs are KV cache math errors, not model size.

## Status

Scaffolded. Concept notes and the first implementation are in progress; results land in `results/` once the GPU runs are done.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/kvcalc/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
