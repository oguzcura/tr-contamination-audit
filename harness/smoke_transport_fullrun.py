"""Pre-launch transport smoke: 1 tiny call per model on opencode-go.

Verifies keys, endpoints, model names and usage/cost capture BEFORE the
13,200-call full run. Writes results/transport_smoke_2026-08-16.jsonl.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "trmlu-audit", "src"))
from trmlu_audit import core

recs = []
client = core.make_client()
for model in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
    rec = core.chat_record(client, model,
                           "You are a test assistant. Reply with exactly the word OK.",
                           "Say OK.", max_tokens=16)
    recs.append({"model": model, "ok": rec["ok"], "retries": rec["retries"],
                 "error": rec["error"], "usage": rec["usage"],
                 "took_s": rec["took_s"], "content": rec["content"][:40],
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(model, "->", "OK" if rec["ok"] else "FAIL", "err:", rec["error"],
          "| usage keys:", sorted(rec["usage"].keys()))

with open("results/transport_smoke_2026-08-16.jsonl", "a", encoding="utf-8") as fh:
    for r in recs:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("all ok:", all(r["ok"] for r in recs))