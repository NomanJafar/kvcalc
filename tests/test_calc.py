import json

import pytest

from kvcalc.calc import ModelSpec, capacity, kv_bytes_per_token

QWEN15B = {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 1536, "num_attention_heads": 12,
           "num_key_value_heads": 2, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}
LLAMA31_8B = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_attention_heads": 32,
              "num_key_value_heads": 8, "num_hidden_layers": 32, "head_dim": 128}
LLAMA2_7B = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_attention_heads": 32,
             "num_hidden_layers": 32}  # no num_key_value_heads -> MHA
DSV2_LITE = {"architectures": ["DeepseekV2ForCausalLM"], "hidden_size": 2048, "num_attention_heads": 16,
             "num_key_value_heads": 16, "num_hidden_layers": 27, "kv_lora_rank": 512, "qk_rope_head_dim": 64}


def test_spec_from_config_reads_gqa_fields_and_derives_head_dim():
    s = ModelSpec.from_config(QWEN15B, name="qwen2.5-1.5b")
    assert (s.n_layers, s.n_kv_heads, s.head_dim, s.attention) == (28, 2, 128, "gqa")


def test_spec_from_config_treats_missing_kv_heads_as_mha():
    s = ModelSpec.from_config(LLAMA2_7B)
    assert s.n_kv_heads == 32 and s.attention == "mha"


def test_spec_from_config_detects_mla():
    s = ModelSpec.from_config(DSV2_LITE)
    assert s.attention == "mla" and s.mla_latent_dim == 576


def test_kv_bytes_per_token_matches_hand_calculation():
    assert kv_bytes_per_token(ModelSpec.from_config(QWEN15B), dtype_bytes=2) == 2 * 28 * 2 * 128 * 2  # 28,672
    assert kv_bytes_per_token(ModelSpec.from_config(LLAMA31_8B), dtype_bytes=2) == 131_072
    assert kv_bytes_per_token(ModelSpec.from_config(LLAMA2_7B), dtype_bytes=2) == 524_288


def test_kv_bytes_per_token_for_mla_uses_latent_not_heads():
    assert kv_bytes_per_token(ModelSpec.from_config(DSV2_LITE), dtype_bytes=2) == 27 * 576 * 2


def test_capacity_splits_budget_into_weights_overhead_and_cache():
    spec = ModelSpec.from_config(QWEN15B, n_params=1.54e9)
    c = capacity(spec, vram_gib=15.0, utilization=0.9, dtype_bytes=2, max_model_len=4096,
                 activation_gib=0.6, overhead_gib=0.4, block_size=16)
    assert c["budget_gib"] == pytest.approx(13.5)
    assert c["weights_gib"] == pytest.approx(1.54e9 * 2 / 2**30, rel=1e-6)
    assert c["kv_budget_gib"] == pytest.approx(13.5 - c["weights_gib"] - 1.0)
    assert c["bytes_per_token"] == 28_672
    assert c["kv_tokens"] == int(c["kv_budget_gib"] * 2**30 // 28_672)
    assert c["kv_blocks"] == c["kv_tokens"] // 16
    assert c["max_concurrency_at_max_len"] == pytest.approx(c["kv_tokens"] / 4096, rel=1e-3)


def test_capacity_reports_shared_prefix_savings():
    spec = ModelSpec.from_config(QWEN15B, n_params=1.54e9)
    base = capacity(spec, vram_gib=15.0, utilization=0.9, dtype_bytes=2, max_model_len=4096,
                    activation_gib=0.6, overhead_gib=0.4)
    shared = capacity(spec, vram_gib=15.0, utilization=0.9, dtype_bytes=2, max_model_len=4096,
                      activation_gib=0.6, overhead_gib=0.4, shared_prefix_tokens=1500)
    assert shared["max_concurrency_at_max_len"] > base["max_concurrency_at_max_len"]
    assert shared["max_concurrency_at_max_len"] == pytest.approx(shared["kv_tokens"] / (4096 - 1500), rel=1e-3)


def test_capacity_refuses_when_weights_exceed_budget():
    spec = ModelSpec.from_config(LLAMA31_8B, n_params=8.03e9)
    c = capacity(spec, vram_gib=15.0, utilization=0.9, dtype_bytes=2, max_model_len=4096)
    assert c["kv_tokens"] == 0 and c["fits"] is False


def test_spec_round_trips_through_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(QWEN15B))
    s = ModelSpec.from_config_file(p)
    assert s.n_layers == 28
