from kvcalc.calc import ModelSpec
from kvcalc.table import KNOWN_MODELS, comparison_table, render_markdown


def test_known_models_cover_the_three_attention_families():
    kinds = {ModelSpec.from_config(cfg, name=n, n_params=p).attention for n, (cfg, p) in KNOWN_MODELS.items()}
    assert kinds == {"mha", "gqa", "mla"}
    assert "Qwen/Qwen2.5-1.5B-Instruct" in KNOWN_MODELS


def test_comparison_table_has_one_row_per_model_and_gpu():
    rows = comparison_table(gpus_gib=[16, 24, 80], max_model_len=4096)
    assert len(rows) == len(KNOWN_MODELS) * 3
    r = next(r for r in rows if r["model"] == "Qwen/Qwen2.5-1.5B-Instruct" and r["vram_gib"] == 16)
    assert r["bytes_per_token"] == 28_672 and r["max_concurrency_at_max_len"] > 50


def test_render_markdown_pivots_gpus_into_columns():
    md = render_markdown(comparison_table(gpus_gib=[16, 80], max_model_len=4096))
    header = md.splitlines()[0]
    assert "16 GB" in header and "80 GB" in header and "KB/token" in header
    assert md.count("\n| ") == len(KNOWN_MODELS)  # one row per model under the header/separator
