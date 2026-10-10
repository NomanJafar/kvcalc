"""`kvcalc predict` (one model, one GPU), `kvcalc table` (known models x GPUs), `kvcalc monitor` (live)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from kvcalc.calc import ModelSpec, capacity
from kvcalc.table import KNOWN_MODELS, comparison_table, render_markdown


def _load_spec(a) -> ModelSpec:
    n_params = float(a.params) if a.params else None
    if a.config:
        return ModelSpec.from_config_file(a.config, name=a.name or Path(a.config).parent.name, n_params=n_params)
    if a.hf:
        if a.hf in KNOWN_MODELS and n_params is None:
            n_params = KNOWN_MODELS[a.hf][1]
        url = f"https://huggingface.co/{a.hf}/resolve/main/config.json"
        cfg = httpx.get(url, follow_redirects=True, timeout=30).raise_for_status().json()
        return ModelSpec.from_config(cfg, name=a.hf, n_params=n_params)
    raise SystemExit("give --config <config.json> or --hf <org/model>")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvcalc", description="KV cache VRAM calculator and monitor for vLLM")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("predict", help="predict KV cache capacity for one model on one GPU")
    s.add_argument("--config", help="path to a HF config.json")
    s.add_argument("--hf", help="HF repo id, fetches config.json (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    s.add_argument("--name")
    s.add_argument("--params", help="parameter count, e.g. 1.54e9 (needed for the weights term)")
    s.add_argument("--vram", type=float, required=True, help="GPU memory in GiB as the driver reports it")
    s.add_argument("--utilization", type=float, default=0.9)
    s.add_argument("--dtype-bytes", type=int, default=2)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--activation-gib", type=float, default=0.6)
    s.add_argument("--overhead-gib", type=float, default=0.4)
    s.add_argument("--block-size", type=int, default=16)
    s.add_argument("--shared-prefix", type=int, default=0, help="tokens of system prompt shared via prefix caching")
    s.add_argument("--json", action="store_true")

    t = sub.add_parser("table", help="compare known models across GPU sizes")
    t.add_argument("--gpus", default="16,24,80", help="GiB values, comma separated")
    t.add_argument("--max-model-len", type=int, default=4096)
    t.add_argument("--json", action="store_true")

    m = sub.add_parser("monitor", help="poll a vLLM /metrics endpoint and compare with the prediction")
    m.add_argument("--base-url", default="http://127.0.0.1:8000")
    m.add_argument("--kv-tokens", type=int, help="predicted capacity in tokens (from `predict`)")
    m.add_argument("--interval", type=float, default=1.0)
    m.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl-C")
    m.add_argument("--out", type=Path, help="write the sampled timeline as JSON")
    return p


def _print_capacity(c: dict) -> None:
    print(f"model: {c['model']}  attention: {c['attention']}  dtype: {c['dtype_bytes']} B")
    print(f"GPU budget: {c['vram_gib']:.1f} GiB x {c['utilization']} = {c['budget_gib']:.2f} GiB")
    print(f"  weights      {c['weights_gib']:.2f} GiB")
    print(f"  activations  {c['activation_gib']:.2f} GiB (assumed)")
    print(f"  overhead     {c['overhead_gib']:.2f} GiB (assumed)")
    print(f"  KV cache     {c['kv_budget_gib']:.2f} GiB")
    print(f"KV cache: {c['bytes_per_token'] / 1024:.1f} KB/token -> {c['kv_tokens']:,} tokens "
          f"({c['kv_blocks']:,} blocks of {c['block_size']})")
    if c["fits"]:
        extra = f" (with {c['shared_prefix_tokens']} shared prefix tokens)" if c["shared_prefix_tokens"] else ""
        print(f"max concurrent requests at {c['max_model_len']} tokens each: "
              f"{c['max_concurrency_at_max_len']:.1f}{extra}")
    else:
        print("model does not fit: weights + overhead exceed the budget")


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "predict":
        c = capacity(_load_spec(a), vram_gib=a.vram, utilization=a.utilization, dtype_bytes=a.dtype_bytes,
                     max_model_len=a.max_model_len, activation_gib=a.activation_gib, overhead_gib=a.overhead_gib,
                     block_size=a.block_size, shared_prefix_tokens=a.shared_prefix)
        print(json.dumps(c, indent=1) if a.json else "", end="")
        if not a.json:
            _print_capacity(c)
    elif a.cmd == "table":
        rows = comparison_table([float(g) for g in a.gpus.split(",")], max_model_len=a.max_model_len)
        print(json.dumps(rows, indent=1) if a.json else render_markdown(rows))
    elif a.cmd == "monitor":
        from kvcalc.monitor import run_monitor
        run_monitor(a.base_url, predicted_tokens=a.kv_tokens, interval_s=a.interval,
                    duration_s=a.duration, out=a.out)


if __name__ == "__main__":
    main()
