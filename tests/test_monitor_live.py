"""Monitor against tokenbench's mock server: proves the polling loop and the
series mapping work end to end, not just the parsing."""

import asyncio
import contextlib
import socket
import threading
import time

import httpx
import uvicorn
from tokenbench.mockserver import MockConfig, create_app

from kvcalc.monitor import run_monitor


@contextlib.contextmanager
def live_mock(**cfg):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(MockConfig(**cfg)), host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.02)
    try:
        yield url
    finally:
        server.should_exit = True
        th.join(timeout=5)


def test_monitor_sees_running_requests_and_token_rate(tmp_path):
    async def load(url):
        async with httpx.AsyncClient(base_url=url, timeout=None) as c:
            body = {"model": "mock", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 40,
                    "stream": True, "ignore_eos": True}
            await asyncio.gather(*[c.post("/v1/chat/completions", json=body) for _ in range(3)])

    with live_mock(ttft_ms=20, itl_ms=10, capacity=2) as url:
        t = threading.Thread(target=lambda: asyncio.run(load(url)), daemon=True)
        t.start()
        rows = run_monitor(url, predicted_tokens=1000, interval_s=0.1, duration_s=0.8, out=tmp_path / "tl.json")
        t.join(timeout=5)
    assert len(rows) >= 5
    assert max(r["running"] for r in rows) == 2          # capacity cap visible
    assert max(r["waiting"] for r in rows) >= 1          # third request queued
    assert any(r["gen_tokens_per_s"] and r["gen_tokens_per_s"] > 0 for r in rows)
    assert any(r["kv_tokens_in_use"] == 1000 for r in rows)  # mock reports usage = running/capacity = 1.0
    assert (tmp_path / "tl.json").exists()
