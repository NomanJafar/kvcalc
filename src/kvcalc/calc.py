"""KV cache arithmetic. See CONCEPTS.md sections 2-3 and 5-6.

Pure functions: a model spec in, bytes per token and capacity out. Nothing here
touches a GPU; the monitor compares these predictions with a live server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    attention: str                    # "mha" | "gqa" | "mla"
    n_params: float | None = None     # parameter count; needed for the weights term
    mla_latent_dim: int | None = None  # kv_lora_rank + qk_rope_head_dim for MLA models

    @classmethod
    def from_config(cls, cfg: dict, name: str | None = None, n_params: float | None = None) -> ModelSpec:
        n_heads = cfg["num_attention_heads"]
        n_kv = cfg.get("num_key_value_heads", n_heads)
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
        if "kv_lora_rank" in cfg:
            latent = cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)
            attention = "mla"
        else:
            latent = None
            attention = "mha" if n_kv == n_heads else "gqa"
        return cls(name=name or cfg.get("_name_or_path", "model"), n_layers=cfg["num_hidden_layers"],
                   n_kv_heads=n_kv, head_dim=head_dim, attention=attention, n_params=n_params,
                   mla_latent_dim=latent)

    @classmethod
    def from_config_file(cls, path: str | Path, **kw) -> ModelSpec:
        return cls.from_config(json.loads(Path(path).read_text()), **kw)


def kv_bytes_per_token(spec: ModelSpec, dtype_bytes: int = 2) -> int:
    if spec.attention == "mla":
        # One compressed latent per token per layer instead of K and V per head.
        return spec.n_layers * spec.mla_latent_dim * dtype_bytes
    return 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * dtype_bytes


def capacity(
    spec: ModelSpec,
    vram_gib: float,
    utilization: float = 0.9,
    dtype_bytes: int = 2,
    max_model_len: int = 4096,
    activation_gib: float = 0.6,
    overhead_gib: float = 0.4,
    block_size: int = 16,
    shared_prefix_tokens: int = 0,
) -> dict:
    """Split the GPU budget the way vLLM does at startup and derive cache capacity.

    activation_gib and overhead_gib are the fuzzy terms (profiled activation peak,
    CUDA context + allocator slack). Defaults are calibrated on a T4; the GPU
    experiment in this project reports how far off they are.
    """
    budget = vram_gib * utilization
    weights = (spec.n_params or 0) * dtype_bytes / GIB
    kv_budget = budget - weights - activation_gib - overhead_gib
    bpt = kv_bytes_per_token(spec, dtype_bytes)
    kv_tokens = int(kv_budget * GIB // bpt) if kv_budget > 0 else 0
    per_request = max(1, max_model_len - shared_prefix_tokens)
    return {
        "model": spec.name,
        "attention": spec.attention,
        "dtype_bytes": dtype_bytes,
        "vram_gib": vram_gib,
        "utilization": utilization,
        "budget_gib": budget,
        "weights_gib": weights,
        "activation_gib": activation_gib,
        "overhead_gib": overhead_gib,
        "kv_budget_gib": max(0.0, kv_budget),
        "bytes_per_token": bpt,
        "kv_tokens": kv_tokens,
        "kv_blocks": kv_tokens // block_size,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "shared_prefix_tokens": shared_prefix_tokens,
        "max_concurrency_at_max_len": kv_tokens / per_request,
        "fits": kv_tokens > 0,
    }
