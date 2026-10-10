import json

from kvcalc.monitor import parse_metrics, snapshot_row, write_timeline

SAMPLE = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
vllm:kv_cache_usage_perc{model_name="m"} 0.25
vllm:num_requests_running{model_name="m"} 12.0
vllm:num_requests_waiting{model_name="m"} 0.0
vllm:num_preemptions_total{model_name="m"} 3.0
vllm:generation_tokens_total{model_name="m"} 50000.0
"""


def test_parse_metrics_strips_labels_and_skips_comments():
    m = parse_metrics(SAMPLE)
    assert m["vllm:kv_cache_usage_perc"] == 0.25 and m["vllm:num_preemptions_total"] == 3.0


def test_snapshot_row_converts_usage_to_tokens_against_prediction():
    row = snapshot_row(parse_metrics(SAMPLE), t=1.5, predicted_tokens=360_000, prev=None)
    assert row["t"] == 1.5 and row["kv_usage"] == 0.25
    assert row["kv_tokens_in_use"] == 90_000
    assert row["running"] == 12 and row["waiting"] == 0 and row["preemptions"] == 3
    assert row["gen_tokens_per_s"] is None  # no previous sample to difference against


def test_snapshot_row_derives_token_rate_from_previous_sample():
    prev = snapshot_row(parse_metrics(SAMPLE), t=0.0, predicted_tokens=None, prev=None)
    later = SAMPLE.replace("50000.0", "50500.0")
    row = snapshot_row(parse_metrics(later), t=2.0, predicted_tokens=None, prev=prev)
    assert row["gen_tokens_per_s"] == 250.0
    assert row["kv_tokens_in_use"] is None


def test_snapshot_row_falls_back_to_old_cache_metric_name():
    old = 'vllm:gpu_cache_usage_perc{m="x"} 0.5\n'
    assert snapshot_row(parse_metrics(old), t=0, predicted_tokens=100, prev=None)["kv_usage"] == 0.5


def test_write_timeline_round_trips(tmp_path):
    rows = [snapshot_row(parse_metrics(SAMPLE), t=0.0, predicted_tokens=100, prev=None)]
    p = tmp_path / "tl.json"
    write_timeline(rows, p, predicted_tokens=100)
    data = json.loads(p.read_text())
    assert data["predicted_tokens"] == 100 and data["samples"][0]["kv_usage"] == 0.25
