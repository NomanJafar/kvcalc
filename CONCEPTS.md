# Concepts: the KV cache is the memory budget

`kvcalc` answers one question before a server is launched: **how many tokens of context can this GPU hold for this model at this precision, and therefore how many concurrent requests?** Getting this wrong is the most common reason an inference server dies in production. The model loaded fine; the cache filled at 3 a.m.

## 1. Where GPU memory goes

For a serving process, VRAM is spent on four things:

| Item | Size | Fixed or variable |
|---|---|---|
| Model weights | parameters × bytes per parameter (3.1 GB for 1.5B in FP16, 16 GB for 8B) | fixed |
| KV cache | grows with every token of every in-flight request | **variable, dominant under load** |
| Activations / workspace | peak intermediate tensors for the largest batch the scheduler will run; profiled once at startup | fixed after profiling |
| CUDA context, allocator overhead, graphs | 0.5–2 GB | fixed |

vLLM's startup does exactly this accounting. `--gpu-memory-utilization 0.9` means "you may use 90% of this GPU's total memory". From that budget it subtracts weights and the profiled activation peak, and everything left becomes KV cache blocks. So the flag does not reserve memory *for* the cache; the cache gets the *remainder*. A model that barely fits leaves almost no cache and therefore almost no concurrency, which is the real reason "it loads but falls over under load".

## 2. Bytes per token

Every transformer layer stores, for each token, one key vector and one value vector per KV head:

```
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
                  ↑ K and V
```

`head_dim` is usually `hidden_size / num_attention_heads`, but read it from `config.json` when present (`head_dim`), because some models decouple it.

Three attention designs change `n_kv_heads`, and they are the whole story of why models differ 10× in cache cost:

- **MHA** (multi-head attention): `n_kv_heads = n_attention_heads`. Llama-2-7B: 32 layers × 32 heads × 128 × 2 bytes × 2 = **524 KB/token**. 4,000 tokens of context cost 2 GB.
- **GQA** (grouped-query attention): several query heads share one KV head. Llama-3.1-8B: 32 × 8 × 128 × 2 × 2 = **131 KB/token** (4× cheaper). Qwen2.5-7B uses 4 KV heads: 28 × 4 × 128 × 2 × 2 = **57 KB/token**. Qwen2.5-1.5B uses 2: **28 KB/token**.
- **MLA** (multi-head latent attention, DeepSeek-V2/V3): the cache stores a compressed latent vector per token instead of K and V. Per layer: `kv_lora_rank + qk_rope_head_dim` elements (512 + 64 = 576 for DeepSeek-V2-Lite), not `2 × heads × head_dim`. DeepSeek-V2-Lite: 27 × 576 × 2 bytes = **31 KB/token** for a 16B-parameter model, which is why MLA matters.

`bytes_per_element` is 2 for FP16/BF16. The KV cache can be quantized separately from the weights (`--kv-cache-dtype fp8` halves it) but that is out of scope here and does not work on a T4.

## 3. From bytes per token to concurrency

```
kv_budget_bytes   = total_vram × gpu_memory_utilization − weights − activation_peak − overhead
max_tokens_cached = kv_budget_bytes / bytes_per_token
max_concurrency   = max_tokens_cached / tokens_per_request
```

where `tokens_per_request` is prompt + generated tokens for a *typical* request, not `max_model_len`. vLLM's startup line "Maximum concurrency for 4,096 tokens per request: 85.65x" is this formula evaluated at `max_model_len`; it is a floor, since most requests are shorter.

Worked example, Qwen2.5-1.5B-Instruct on a 16 GB T4 (15.0 GiB usable), FP16:

```
budget      = 15.0 × 0.9        = 13.5 GiB
weights     = 1.54B × 2 bytes   = 2.9 GiB
activations ≈ 0.6 GiB (profiled; depends on max-num-batched-tokens)
overhead    ≈ 0.4 GiB
kv_budget   ≈ 9.6 GiB
tokens      = 9.6 GiB / 28 KB   ≈ 360,000 tokens
at 4,096 tokens/request: ~88 concurrent   |   at 1,000: ~360
```

The notebook in project 1 asks vLLM for its real numbers; the calculator's job is to predict them within a few percent *before* paying for a GPU. The activation and overhead terms are the fuzzy part: the calculator exposes them as parameters with defaults calibrated on the T4 run, and the README reports the prediction error.

## 4. Why servers OOM anyway

If vLLM pre-allocates the cache, how does anything run out of memory? Several ways, each a different lesson:

1. **`gpu-memory-utilization` set too high on a shared GPU.** Another process (a second model, a notebook, the display) holds memory; vLLM's budget assumes it has the whole fraction. Fails at startup or, worse, at the first large batch.
2. **Activation peak larger than profiled.** The profile run uses `max-num-batched-tokens` and `max-num-seqs`; a feature enabled later (a LoRA adapter, a different sampler, speculative decoding) raises the real peak above it.
3. **Memory fragmentation outside the paged cache.** The weights and cache are fine; the PyTorch allocator's free blocks are fragmented by varying activation shapes. Symptom: `CUDA out of memory` with hundreds of MB "reserved but unallocated".
4. **Someone changed `max-model-len` without recomputing.** Longer context = fewer concurrent requests, not an OOM, but the queue grows and TTFT explodes, and people reach for `gpu-memory-utilization 0.98`, which causes case 1.
5. **The host side.** Swap space for preempted sequences (`--swap-space`, CPU RAM) and pinned-memory buffers can exhaust host RAM.

The calculator addresses 1 and 4 directly (it tells you the concurrency consequence of a config change). The monitor addresses the rest by showing the cache fill level and preemption count live, so the OOM-adjacent symptoms are visible before the crash.

## 5. Blocks, not tokens

vLLM allocates the cache in **blocks** of `block_size` tokens (16 by default). A request holding 1,000 tokens uses 63 blocks (the last one partially filled); the waste is at most `block_size − 1` tokens per sequence, which is why block size is small. The `/metrics` series `vllm:kv_cache_usage_perc` is blocks in use over blocks total. The calculator reports block counts too, so its numbers line up with what the server prints and exports.

Block size trades waste against attention-kernel efficiency; project 9 measures that trade. Here it only affects rounding.

## 6. Prefix caching changes the accounting

With `--enable-prefix-caching`, blocks whose content hash matches an earlier request's prefix are shared, not duplicated. Memory per request drops by the shared prefix length. The calculator takes an optional "shared prefix tokens" parameter so a chat deployment with a 1,500-token system prompt can see what sharing buys (project 4 measures it).

## 7. What the live monitor shows

Polling `/metrics` once a second and printing:

- `kv_cache_usage` (fraction), converted back to tokens using the calculator's bytes/token, next to the predicted capacity;
- `num_requests_running` and `num_requests_waiting`: waiting > 0 while cache usage < 100% means the limit is `max-num-seqs`, not memory;
- `num_preemptions_total` if exported: a preemption means a running request was evicted to make room; sustained preemptions mean the cache is undersized for the load;
- tokens per second, so you can see throughput fall when preemptions start.

The point of showing predicted and actual side by side is calibration: if the prediction is off by 20%, the activation/overhead assumptions are wrong for this model or version, and the calculator should be corrected, not the server flag.

## 8. What the experiment will show

1. Predict KV cache size for Qwen2.5-1.5B at `max-model-len` 2048 and 8192 on the T4; launch vLLM at both; compare predicted vs reported "GPU KV cache size: N tokens" and "Available KV cache memory". Report the error.
2. Run the monitor while a `tokenbench` closed-loop sweep climbs to the point where `num_requests_waiting` rises; confirm whether the cap was memory or `max-num-seqs`.
3. Comparison table for five models (Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-1.5B, Mistral-7B, DeepSeek-V2-Lite): bytes/token and max concurrent 4k-token requests on 16 GB, 24 GB and 80 GB GPUs in FP16.

## Further reading

- vLLM docs: *Engine Arguments* (`gpu-memory-utilization`, `max-model-len`, `block-size`, `swap-space`) and the startup log format.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023).
- DeepSeek-AI, *DeepSeek-V2* (2024), section on Multi-head Latent Attention.
- Kwon et al., *PagedAttention* (SOSP 2023), section 4, for the block design.
