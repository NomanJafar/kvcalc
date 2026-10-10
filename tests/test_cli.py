import json

from kvcalc.cli import main

QWEN = {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 1536, "num_attention_heads": 12,
        "num_key_value_heads": 2, "num_hidden_layers": 28}


def test_predict_from_config_file_prints_capacity(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(QWEN))
    main(["predict", "--config", str(p), "--params", "1.54e9", "--vram", "15", "--max-model-len", "4096", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["bytes_per_token"] == 28_672 and out["kv_tokens"] > 300_000


def test_predict_human_output_mentions_tokens_and_concurrency(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(QWEN))
    main(["predict", "--config", str(p), "--params", "1.54e9", "--vram", "15"])
    text = capsys.readouterr().out
    assert "tokens" in text and "concurrent" in text


def test_table_subcommand_prints_markdown(capsys):
    main(["table", "--gpus", "16,80"])
    assert "| model" in capsys.readouterr().out.lower()
