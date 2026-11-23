from __future__ import annotations

import json
import sys

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
question = sys.argv[2] if len(sys.argv) > 2 else "Чем отличается git merge от git rebase?"

with httpx.Client(base_url=API, timeout=180) as client:
    health = client.get("/health")
    print("health", health.status_code, health.json())
    ready = client.get("/ready")
    print("ready", ready.status_code, ready.json())
    response = client.post("/chat", json={"question": question, "top_k": 6})
    print("chat", response.status_code)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
