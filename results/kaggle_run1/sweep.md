**tokenbench run** · model: Qwen/Qwen2.5-1.5B-Instruct · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 56 | 41 ms | 44 ms | 17.8 ms | 20.3 ms | 18.4 ms |
| 4 | 64 | 0 | 207 | 72 ms | 77 ms | 19.1 ms | 20.2 ms | 19.6 ms |
| 16 | 64 | 0 | 522 | 157 ms | 213 ms | 30.0 ms | 32.1 ms | 30.8 ms |
| 64 | 64 | 0 | 853 | 411 ms | 662 ms | 73.3 ms | 79.0 ms | 75.2 ms |
| 128 | 64 | 0 | 851 | 406 ms | 608 ms | 73.6 ms | 79.4 ms | 75.5 ms |
