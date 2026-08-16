"""Core implementation of the TR-MMLU audit probes.

API-only (black-box) contamination detection for TR-MMLU
(alibayram/turkish_mmlu). No weights, no GPU; works on hosted models.

Probes implemented:
  - Black Box           : forward-pass accuracy (baseline)
  - Choice substitution : Yao et al. (2024) pondered distractor planting
  - Cross-lingual       : Tr->En->Tr back-translation + Tr->En direct arms

TRANSPORT (2026-08-16, Day 2): all inference now runs through **opencode-go**
(https://opencode.ai/zen/go/v1, key OPENCODE_GO_API_KEY). OpenRouter is blocked
for this account (404 "no allowed providers"). Live-verified pitfalls:
  (a) raw urllib/httpx calls get Cloudflare HTTP 403 unless a browser-like
      User-Agent is sent; the OpenAI SDK path works as-is.
  (b) mimo-v2.5 intermittently returns an empty/None message -> retry-on-empty
      (max 2 retries), each failure logged.
  (c) reasoning models (deepseek-v4-flash) put chain-of-thought in
      `reasoning_content` -> parse content + reasoning_content; keep the
      `parse_letter` verbosity handling.
"""
from __future__ import annotations
import json
import os
import random
import re
import time
from typing import Dict, List, Optional

TRANS_MODEL = "deepseek-v4-flash"
ANS_MODEL = "deepseek-v4-flash"
_SEED = 42

DOTENV = os.path.expanduser(r"C:\Users\oguzc\AppData\Local\hermes\.env")
OPENCODE_BASE = "https://opencode.ai/zen/go/v1"
MAX_EMPTY_RETRIES = 2          # (b) retry-on-empty, per task spec

# ---------------------------------------------------------------------------
# Client / transport  (opencode-go; OpenRouter NOT used)
# ---------------------------------------------------------------------------
def load_secret(name: str, env_file: str = DOTENV) -> Optional[str]:
    """Env var first, then the hermes .env file. Never echoes the value."""
    v = os.getenv(name)
    if v:
        return v
    try:
        for line in open(env_file, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def make_client():
    """OpenAI-SDK client pointed at opencode-go (the only allowed transport)."""
    key = load_secret("OPENCODE_GO_API_KEY")
    if not key:
        raise RuntimeError("OPENCODE_GO_API_KEY not found (env or %s)" % DOTENV)
    from openai import OpenAI
    return OpenAI(
        base_url=os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_BASE),
        api_key=key,
    )


def _message_text(msg, fallback_tail: bool = True) -> str:
    """content + reasoning_content fallbacks, per pitfall (c)."""
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content
    # OpenAI-SDK surfaces reasoning via `reasoning_content` (or model_extra),
    # and older transports via `reasoning`.
    reason = getattr(msg, "reasoning_content", None)
    if not reason:
        reason = (msg.model_extra or {}).get("reasoning_content") if hasattr(msg, "model_extra") else None
    if not reason:
        reason = getattr(msg, "reasoning", None)
    if reason and fallback_tail:
        return str(reason)[-200:]
    return ""


def _usage_dict(usage) -> dict:
    """Capture every usage/cost field the API returned (never guess)."""
    if usage is None:
        return {}
    out = {}

    def _conv(v):
        if hasattr(v, "__dict__"):            # SDK objects -> nested dicts
            return {k: _conv(x) for k, x in vars(v).items()}
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_conv(x) for x in v]
        try:
            json.dumps(v)
            return v
        except TypeError:
            return str(v)

    try:
        for k, v in vars(usage).items():
            out[k] = _conv(v)
    except TypeError:
        out["_raw"] = str(usage)
    return out


def _chat_once(client, model: str, system: str, user: str,
               max_tokens: int = 1600, fallback_tail: bool = True):
    """One raw call. Returns (content, usage_dict, error_or_None)."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0)
        msg = r.choices[0].message
        return _message_text(msg, fallback_tail), _usage_dict(getattr(r, "usage", None)), None
    except Exception as exc:                      # (a) 403 / network / provider
        return "", {}, f"{type(exc).__name__}: {exc}"


def chat_record(client, model: str, system: str, user: str,
                max_tokens: int = 1600, fallback_tail: bool = True) -> dict:
    """Harness call path with retry-on-empty (b). Returns a log-ready record.

    Record fields: model, ok, content, retries, error, usage, took_s.
    """
    retries = 0
    err = None
    usage = {}
    content = ""
    t0 = time.time()
    for attempt in range(1 + MAX_EMPTY_RETRIES):
        content, usage, err = _chat_once(client, model, system, user,
                                         max_tokens, fallback_tail)
        if content:                               # non-empty answer -> done
            return {"model": model, "ok": True, "content": content,
                    "retries": retries, "error": None, "usage": usage,
                    "took_s": round(time.time() - t0, 2)}
        if attempt < MAX_EMPTY_RETRIES:
            retries += 1
            print(f"[retry] {model} empty/error on attempt {attempt+1}: {err or 'empty message'}")
            time.sleep(1.5 * (attempt + 1))
    return {"model": model, "ok": False, "content": content,
            "retries": retries,
            "error": err or "empty message after %d retries" % (MAX_EMPTY_RETRIES + 1),
            "usage": usage, "took_s": round(time.time() - t0, 2)}


def _chat(client, model: str, system: str, user: str,
          max_tokens: int = 1600, fallback_tail: bool = True) -> str:
    """Compatibility wrapper (existing probes): return content string only."""
    return chat_record(client, model, system, user,
                       max_tokens, fallback_tail)["content"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_tr_mmlu(split: str = "mmlu", limit: Optional[int] = None,
                 seed: int = _SEED) -> List[Dict]:
    """Load the gated TR-MMLU split. Requires HF_TOKEN for first download."""
    if not os.getenv("HF_TOKEN"):
        import sys
        print("[!] HF_TOKEN not set. TR-MMLU is gated (CC BY-NC-ND 4.0).",
              file=sys.stderr)
    from datasets import load_dataset
    rows = list(load_dataset("alibayram/turkish_mmlu", split=split))
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def idx_to_letter(idx) -> str:
    """TR-MMLU stores 'cevap' as an integer index into secenekler."""
    try:
        return chr(65 + int(idx))
    except (TypeError, ValueError):
        return "?"


def item_text(row: Dict) -> str:
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["secenekler"])))


# ---------------------------------------------------------------------------
# Answering + parsing
# ---------------------------------------------------------------------------
def answer_item(client, text: str, lang_prompt: str) -> str:
    return _chat(client, ANS_MODEL,
                 "You are taking a standardized multiple-choice exam. " + lang_prompt,
                 text + "\n\nCevap (single letter A/B/C/D/E):")


def parse_letter(raw: Optional[str]) -> Optional[str]:
    """Extract A-E answer letter from free-form model output (incl. reasoning)."""
    if not raw:
        return None
    raw = raw.strip()
    for pat in (r'\b([A-Ea-e])\b\s*\)', r'\bCevap:\s*([A-Ea-e])\b',
                r'\b([A-Ea-e])\b'):
        m = re.search(pat, raw)
        if m and m.group(1).upper() in "ABCDE":
            return m.group(1).upper()
    m = re.fullmatch(r'\s*([A-Ea-e])\s*', raw)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def run_black_box(client, rows: List[Dict], limit: int) -> List[Dict]:
    out = []
    for i, row in enumerate(rows[:limit]):
        out.append({"i": i, "bolum": row["bolum"], "type": "blackbox",
                    "truth": idx_to_letter(row["cevap"]),
                    "raw": answer_item(client, item_text(row),
                                       "Answer in Turkish (the question is in Turkish). ")})
    return out


def build_perturbed(rows: List[Dict], idx: int) -> List[str]:
    """Replace one distractor with the correct answer of another question."""
    correct = rows[idx]["secenekler"][int(rows[idx]["cevap"])]
    other = rows[(idx + 1) % len(rows)]["secenekler"][int(rows[(idx + 1) % len(rows)]["cevap"])]
    pert = list(rows[idx]["secenekler"])
    for i, c in enumerate(pert):
        if c != correct:
            pert[i] = other
            break
    return pert


def run_choice_substitution(client, rows: List[Dict], limit: int) -> List[Dict]:
    out = []
    item_text_pert = item_text
    for i in range(min(limit, len(rows))):
        base = ("Soru: " + rows[i]["soru"] + "\n\nSecenekler:\n" +
                "\n".join(f"{chr(65+k)}) {c}" for k, c in
                          enumerate(build_perturbed(rows, i))))
        out.append({"i": i, "bolum": rows[i]["bolum"], "type": "choice_sub",
                    "truth": idx_to_letter(rows[i]["cevap"]),
                    "raw": answer_item(client, base,
                                       "Answer in Turkish (the question is in Turkish). ")})
    return out


def tr_to_en(client, row: Dict) -> str:
    sys_ = ("You are a faithful professional Turkish->English translator. Translate the "
            "whole question AND every choice. Keep letter labels (A,B,C,..) as-is. "
            "Preserve meaning precisely. Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, item_text(row))


def en_to_tr(client, en_text: str) -> str:
    sys_ = ("You are a faithful professional English->Turkish translator. Translate back to "
            "Turkish. Keep letter labels (A,B,C,..) as-is. Preserve meaning precisely. "
            "Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, en_text)


def roundtrip(client, row: Dict) -> str:
    return en_to_tr(client, tr_to_en(client, row))


def run_crosslingual(client, rows: List[Dict], limit: int,
                     arms: Optional[List[str]] = None) -> List[Dict]:
    arms = arms or ["B", "C"]
    out = []
    for i, row in enumerate(rows[:limit]):
        rec = {"i": i, "bolum": row["bolum"], "truth": idx_to_letter(row["cevap"]),
               "soru": row["soru"], "secenekler": row["secenekler"]}
        if "B" in arms:
            rec["B_backtr"] = roundtrip(client, row)
            rec["B_answer"] = answer_item(
                client, rec["B_backtr"],
                "Answer in Turkish (the question is in Turkish). ")
        if "C" in arms:
            en = tr_to_en(client, row)
            rec["C_en"] = en
            rec["C_answer"] = answer_item(
                client, en, "Answer in English (the item is in English). ")
        out.append(rec)
    return out