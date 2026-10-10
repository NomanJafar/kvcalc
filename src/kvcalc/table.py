"""Comparison table across well-known models and GPU sizes.

KNOWN_MODELS holds the attention-relevant fields copied from each model's
config.json on the Hugging Face Hub plus its parameter count. `kvcalc predict --hf`
fetches a live config instead; this table exists so the README has numbers
without network access.
"""

from __future__ import annotations

from kvcalc.calc import ModelSpec, capacity, kv_bytes_per_token

# name -> (config.json subset, parameter count)
KNOWN_MODELS: dict[str, tuple[dict, float]] = {
    "meta-llama/Llama-2-7b-hf": ({"hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 32}, 6.74e9),
    "meta-llama/Llama-3.1-8B": ({"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
                                 "num_hidden_layers": 32}, 8.03e9),
    "mistralai/Mistral-7B-v0.3": ({"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
                                   "num_hidden_layers": 32}, 7.25e9),
    "Qwen/Qwen2.5-7B-Instruct": ({"hidden_size": 3584, "num_attention_heads": 28, "num_key_value_heads": 4,
                                  "num_hidden_layers": 28}, 7.62e9),
    "Qwen/Qwen2.5-1.5B-Instruct": ({"hidden_size": 1536, "num_attention_heads": 12, "num_key_value_heads": 2,
                                    "num_hidden_layers": 28}, 1.54e9),
    "deepseek-ai/DeepSeek-V2-Lite": ({"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 16,
                                      "num_hidden_layers": 27, "kv_lora_rank": 512, "qk_rope_head_dim": 64}, 15.7e9),
}


def comparison_table(gpus_gib: list[float], max_model_len: int = 4096, utilization: float = 0.9,
                     dtype_bytes: int = 2) -> list[dict]:
    rows = []
    for name, (cfg, n_params) in KNOWN_MODELS.items():
        spec = ModelSpec.from_config(cfg, name=name, n_params=n_params)
        for vram in gpus_gib:
            c = capacity(spec, vram_gib=vram, utilization=utilization, dtype_bytes=dtype_bytes,
                         max_model_len=max_model_len)
            rows.append({"model": name, "attention": spec.attention, "n_kv_heads": spec.n_kv_heads,
                         "vram_gib": vram, "bytes_per_token": kv_bytes_per_token(spec, dtype_bytes),
                         "weights_gib": c["weights_gib"], "kv_tokens": c["kv_tokens"],
                         "max_concurrency_at_max_len": c["max_concurrency_at_max_len"], "fits": c["fits"]})
    return rows


def render_markdown(rows: list[dict]) -> str:
    gpus = sorted({r["vram_gib"] for r in rows})
    head = ["model", "attn", "KV heads", "KB/token"] + [f"{g:g} GB: 4k-ctx concurrency" for g in gpus]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in dict.fromkeys(r["model"] for r in rows):
        rs = {r["vram_gib"]: r for r in rows if r["model"] == name}
        any_r = next(iter(rs.values()))
        cells = [name, any_r["attention"].upper(), str(any_r["n_kv_heads"]), f"{any_r['bytes_per_token'] / 1024:.0f}"]
        for g in gpus:
            r = rs[g]
            c = r["max_concurrency_at_max_len"]
            cells.append("does not fit" if not r["fits"] else f"{c:.1f}" if c < 10 else f"{c:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
