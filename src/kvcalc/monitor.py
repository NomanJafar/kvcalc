"""Live KV cache monitor: poll vLLM's /metrics and show usage against the prediction."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")

_SERIES = {
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "preemptions": ("vllm:num_preemptions_total",),
    "gen_tokens_total": ("vllm:generation_tokens_total",),
}


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass
    return out


def _pick(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    return next((metrics[n] for n in names if n in metrics), None)


def snapshot_row(metrics: dict[str, float], t: float, predicted_tokens: int | None, prev: dict | None) -> dict:
    row: dict = {"t": t}
    for short, names in _SERIES.items():
        row[short] = _pick(metrics, names)
    usage = row["kv_usage"]
    row["kv_tokens_in_use"] = round(usage * predicted_tokens) if (usage is not None and predicted_tokens) else None
    if prev and row["gen_tokens_total"] is not None and prev.get("gen_tokens_total") is not None and t > prev["t"]:
        row["gen_tokens_per_s"] = (row["gen_tokens_total"] - prev["gen_tokens_total"]) / (t - prev["t"])
    else:
        row["gen_tokens_per_s"] = None
    for k in ("running", "waiting", "preemptions"):
        if row[k] is not None:
            row[k] = int(row[k])
    return row


def write_timeline(rows: list[dict], path: Path | str, predicted_tokens: int | None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"predicted_tokens": predicted_tokens, "samples": rows}, indent=1))


def _fmt(row: dict, predicted: int | None) -> str:
    usage = row["kv_usage"]
    pct = f"{usage * 100:5.1f}%" if usage is not None else "   n/a"
    tok = f"{row['kv_tokens_in_use']:>8,}/{predicted:,} tok" if row["kv_tokens_in_use"] is not None else ""
    rate = f"{row['gen_tokens_per_s']:7.1f} tok/s" if row["gen_tokens_per_s"] is not None else "    -   tok/s"
    return (f"t={row['t']:6.1f}s  kv {pct} {tok}  running={row['running']}  waiting={row['waiting']}  "
            f"preempt={row['preemptions']}  {rate}")


def run_monitor(base_url: str, predicted_tokens: int | None, interval_s: float = 1.0,
                duration_s: float = 0, out: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    prev = None
    t0 = time.monotonic()
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            while True:
                t = time.monotonic() - t0
                try:
                    metrics = parse_metrics(client.get("/metrics").raise_for_status().text)
                    row = snapshot_row(metrics, t, predicted_tokens, prev)
                    rows.append(row)
                    prev = row
                    print(_fmt(row, predicted_tokens), file=sys.stderr)
                except httpx.HTTPError as e:
                    print(f"t={t:6.1f}s  metrics unavailable: {type(e).__name__}", file=sys.stderr)
                if duration_s and t >= duration_s:
                    break
                time.sleep(interval_s)
    except KeyboardInterrupt:
        pass
    if out:
        write_timeline(rows, out, predicted_tokens)
    return rows
