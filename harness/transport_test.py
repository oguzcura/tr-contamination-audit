"""Transport test: opencode-go through the exact harness code path.

One chat call per lineup model via trmlu_audit.core.chat_record
(make_client + chat_record = the path probe scripts will use), plus a
GET /models listing with a browser User-Agent (pitfall a).

Usage:  .venv/Scripts/python.exe transport_test.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "trmlu-audit", "src"))

from trmlu_audit import core  # noqa: E402

LINEUP = ["gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def models_list():
    """GET /models with browser UA (raw-urllib would 403 without it)."""
    import urllib.request
    req = urllib.request.Request(core.OPENCODE_BASE + "/models",
                                 headers={"User-Agent": UA,
                                          "Authorization": "Bearer " + core.load_secret("OPENCODE_GO_API_KEY")})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main():
    print("### /models (browser UA) ###")
    try:
        data = models_list()
        entries = data.get("data") or data.get("models") or []
        print(f"total models listed: {len(entries)}")
        for m in entries:
            mid = m.get("id", "?")
            if mid in LINEUP:
                print(f"  FOUND lineup id: {mid} | pricing={m.get('pricing')} | ctx={m.get('context_length')}")
    except Exception as exc:
        print(f"[warn] /models failed: {type(exc).__name__}: {exc}")

    print("\n### one call per lineup model (exact harness path) ###")
    client = core.make_client()
    for model in LINEUP:
        rec = core.chat_record(
            client, model,
            "You are a terse assistant. Answer with one word only.",
            "Merhaba! 2+2 kac eder? Tek kelimeyle yanitla.",
            max_tokens=50)
        usage = rec["usage"]
        print(f"\nmodel={model} ok={rec['ok']} retries={rec['retries']} "
              f"error={rec['error']} took={rec['took_s']}s")
        print(f"  content={rec['content'][:80]!r}")
        print(f"  usage={json.dumps(usage, ensure_ascii=False)}")
        if not rec["ok"]:
            print(f"  FAILED: {rec['error']}")


if __name__ == "__main__":
    main()
