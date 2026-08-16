"""Probe M1 - verbatim-recall / recognition (surface-exactness). Day 3.

For a sampled item: present the question + the option set with the answer
position ordering destroyed (options are shuffled), then ask the model to
produce the correct answer AND reproduce the exact full text of the correct
choice. The contamination signature (M1, design §3) is 8-gram overlap between
the model's output and the ground-truth option surface, plus the proportion of
items where the model emits the memorized correct option verbatim.

TRANSPORT (2026-08-16): opencode-go via trmlu_audit.core.chat_record
(retry-on-empty max 2, content+reasoning_content parsing, usage capture).
Every call is appended to a JSONL log: model, tokens, API usage/cost fields,
retries, failure flag.

Run:
  uv run python probe_m1.py --limit 20 --model gpt-5.6-luna
                             --log results/smoke_2026-08-16.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "trmlu-audit", "src"))
from trmlu_audit import core  # noqa: E402

SEED = 42
NGRAM = 8

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data: TR-MMLU (primary target; cached locally)
# ---------------------------------------------------------------------------
def load_tr_mmlu(limit: int, seed: int = SEED) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("alibayram/turkish_mmlu", split="mmlu")
    rows = list(ds)
    rng = random.Random(seed)
    return rng.sample(rows, min(limit, len(rows)))


def _as_list(x) -> List[str]:
    if isinstance(x, list):
        return list(x)
    s = str(x).strip().strip("[]()")
    return [c.strip().strip("'\" ").strip() for c in s.split(",") if c.strip()]


def make_m1_prompt(row: Dict, idx: int) -> tuple[str, str, str]:
    """Return (prompt, ground_truth_option_text, truth_shuffled_letter).

    Options are SHUFFLED (deterministic per item) so the answer-position
    ordering is stripped; overlap is measured against the ground-truth
    option surface text only.
    """
    opts = _as_list(row["secenekler"])
    truth_idx = int(row["cevap"])
    truth_text = opts[truth_idx]
    rng = random.Random(SEED + idx)
    order = list(range(len(opts)))
    rng.shuffle(order)
    shuffled = [opts[i] for i in order]
    truth_new_letter = chr(65 + order.index(truth_idx))
    q = str(row["soru"]).strip()
    lines = [f"Soru: {q}", "", "Secenekler:"]
    lines += [f"{chr(65 + i)}) {c}" for i, c in enumerate(shuffled)]
    prompt = "\n".join(lines) + (
        "\n\nCevap (ONLY the letter of the correct choice, e.g. A) followed by "
        "the FULL EXACT text of that choice, reproduced verbatim as it appears above:")
    return prompt, truth_text, truth_new_letter


def make_bb_prompt(row: Dict) -> str:
    """Plain black-box MC prompt (transport smoke, n=5 phase)."""
    opts = _as_list(row["secenekler"])
    lines = [f"Soru: {str(row['soru']).strip()}", "", "Secenekler:"]
    lines += [f"{chr(65 + i)}) {c}" for i, c in enumerate(opts)]
    return "\n".join(lines) + "\n\nCevap (tek harf A/B/C/D/E):"


# ---------------------------------------------------------------------------
# Overlap scorer (design §3 M1)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def grams8(s: str):
    s = _norm(s)
    return {s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1)}


def overlap8(target: str, output: str) -> float:
    """Coverage of target 8-grams by the output (0.0 if target < 8 chars)."""
    tg, og = grams8(target), grams8(output)
    if not tg:
        return 0.0
    return len(tg & og) / len(tg)


def is_verbatim(target: str, output: str) -> bool:
    t, o = _norm(target), _norm(output)
    return t != "" and t in o


# ---------------------------------------------------------------------------
# Cost estimation (documented rates only; None = no published rate)
#   deepseek-v4-flash: spec ladder (paper1_endpoints.md)  in/cache/out per 1M
#   gpt-5.6-luna     : resolved live 2026-08-16 (endpoint notes) $0.10 / $0.60
#   mimo-v2.5        : no published rate -> tokens only (cost_est = None)
# ---------------------------------------------------------------------------
LADDER = {
    "deepseek-v4-flash": (0.07, 0.0014, 0.14),
    "gpt-5.6-luna": (0.10, 0.0, 0.60),
    "mimo-v2.5": None,
}


def cost_estimate(model: str, usage: dict) -> Optional[float]:
    if model not in LADDER or LADDER[model] is None:
        return None
    p_in, p_cache, p_out = LADDER[model]
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    ptd = usage.get("prompt_tokens_details") or {}
    cached = int(ptd.get("cached_tokens") or 0)
    return round(pt * p_in / 1e6 + cached * p_cache / 1e6 + ct * p_out / 1e6, 8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", default="m1_smoke")
    ap.add_argument("--log", default="results/smoke_2026-08-16.jsonl")
    ap.add_argument("--bb", action="store_true",
                    help="plain black-box letter answer (transport smoke) instead of M1")
    args = ap.parse_args()

    rows = load_tr_mmlu(args.limit, seed=SEED)
    client = core.make_client()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    total_pt = total_ct = total_cached = 0
    n_ok = n_fail = n_retries = 0
    cost_total = 0.0
    hits = 0
    ovs: List[float] = []

    with open(args.log, "a", encoding="utf-8") as f:
        for k, row in enumerate(rows):
            if args.bb:
                prompt = make_bb_prompt(row)
                system = ("You are taking a standardized Turkish multiple-choice "
                          "exam. Answer with ONLY the single letter (A/B/C/D/E).")
            else:
                prompt, truth_text, _ = make_m1_prompt(row, k)
                system = ("You are taking a standardized Turkish multiple-choice "
                          "exam. Give ONLY the letter and then the verbatim choice text.")
            rec = core.chat_record(client, args.model, system, prompt,
                                   max_tokens=700)
            usage = rec["usage"]
            raw = rec["content"]
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            ptd = usage.get("prompt_tokens_details") or {}
            cached = int(ptd.get("cached_tokens") or 0)
            ctd = usage.get("completion_tokens_details") or {}
            reason_toks = int(ctd.get("reasoning_tokens") or 0)
            cost = cost_estimate(args.model, usage)

            total_pt += pt
            total_ct += ct
            total_cached += cached
            n_ok += int(rec["ok"])
            n_fail += int(not rec["ok"])
            n_retries += rec["retries"]
            if cost is not None:
                cost_total += cost

            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "phase": args.phase,
                "model": args.model,
                "item_idx": k,
                "bolum": str(row.get("bolum", "")),
                "truth_letter": core.idx_to_letter(row["cevap"]),
                "prompt_tokens": pt,
                "cached_tokens": cached,
                "completion_tokens": ct,
                "reasoning_tokens": reason_toks,
                "total_tokens": pt + ct,
                "usage_api": usage,          # cost fields as returned by API (none today)
                "cost_est_usd": cost,        # documented-rate estimate (None = no rate)
                "retries": rec["retries"],
                "ok": rec["ok"],
                "error": rec["error"],
                "took_s": rec["took_s"],
            }
            if args.bb:
                parsed = core.parse_letter(raw)
                record.update({
                    "raw": raw[:400],
                    "parsed_letter": parsed,
                    "correct": parsed == core.idx_to_letter(row["cevap"]),
                })
            else:
                ov = overlap8(truth_text, raw)
                vb = is_verbatim(truth_text, raw)
                hits += int(vb)
                ovs.append(ov)
                record.update({
                    "truth_shuffled_letter": _truth_new_letter(row, k),
                    "truth_text": truth_text,
                    "raw": raw[:400],
                    "overlap8": round(ov, 4),
                    "verbatim_hit": vb,
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            tag = "BB" if args.bb else "M1"
            print(f"[{tag} {k+1}/{len(rows)}] {args.model} "
                  f"{'ok' if rec['ok'] else 'FAIL'} retries={rec['retries']} "
                  f"tokens={pt}+{ct} cost≈{cost}")

    n = len(rows)
    print(f"\n=== {args.phase} SUMMARY ({args.model}, n={n}) ===")
    print(f"ok={n_ok} fail={n_fail} retries={n_retries}")
    print(f"prompt_tokens={total_pt} cached={total_cached} completion_tokens={total_ct}")
    print(f"cost_est_total=${cost_total:.6f}" if cost_total else
          f"cost_est_total=None (no published rate for {args.model})")
    if not args.bb:
        mean_ov = sum(ovs) / len(ovs) if ovs else 0.0
        print(f"verbatim hits={hits}/{n} mean overlap8={mean_ov:.4f}")


def _truth_new_letter(row: Dict, idx: int) -> str:
    opts = _as_list(row["secenekler"])
    truth_idx = int(row["cevap"])
    rng = random.Random(SEED + idx)
    order = list(range(len(opts)))
    rng.shuffle(order)
    return chr(65 + order.index(truth_idx))


if __name__ == "__main__":
    main()
