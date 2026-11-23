from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

QUESTIONS = [
    "Чем отличается git merge от git rebase?",
    "Как выбрать индекс для составного условия в SQL?",
    "Почему деструктор базового класса должен быть виртуальным?",
    "Как работает трехстороннее рукопожатие TCP?",
    "Что такое динамическое программирование?",
]


async def one_request(client: httpx.AsyncClient, question: str) -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        response = await client.post("/chat", json={"question": question, "top_k": 6})
        response.raise_for_status()
        ok = True
    except Exception:  # noqa: BLE001
        ok = False
    return ok, (time.perf_counter() - started) * 1000


async def run_concurrency(api_url: str, concurrency: int, total: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=api_url, timeout=180) as client:
        async def worker(index: int):
            async with semaphore:
                return await one_request(client, QUESTIONS[index % len(QUESTIONS)])

        started = time.perf_counter()
        results = await asyncio.gather(*(worker(index) for index in range(total)))
        elapsed = time.perf_counter() - started
    latencies = [latency for ok, latency in results if ok]
    errors = sum(1 for ok, _ in results if not ok)
    return {
        "concurrency": concurrency,
        "requests": total,
        "errors": errors,
        "elapsed_s": round(elapsed, 2),
        "rps": round(total / elapsed, 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "mean_ms": round(float(np.mean(latencies)), 1) if latencies else 0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--requests", type=int, default=15)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.api_url, timeout=180) as client:
        for question in QUESTIONS[:2]:
            await one_request(client, question)

    runs = []
    for concurrency in (1, 3):
        result = await run_concurrency(args.api_url, concurrency, args.requests)
        print(result, flush=True)
        runs.append(result)

    primary = runs[0]
    ram_mb = 0
    pid = os.getenv("API_PID")
    if pid:
        try:
            import psutil

            ram_mb = round(psutil.Process(int(pid)).memory_info().rss / 1024 / 1024, 1)
        except Exception:  # noqa: BLE001
            pass

    payload = {
        "runs": runs,
        "p50_ms": primary["p50_ms"],
        "p95_ms": primary["p95_ms"],
        "rps": max(run["rps"] for run in runs),
        "ram_mb": ram_mb,
        "startup_s": float(os.getenv("STARTUP_S", "0") or 0),
        "model": os.getenv("LLM_MODEL", ""),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/performance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
