"""Positive-control experiment for the Turkish contamination audit.

PURPOSE
--------
The main audit reported an honest negative: 0/12 cells flagged under M1-DISCOUNT.
A reviewer's legitimate concern is that the probes LACK POWER --- i.e. they would
not detect contamination even if it were present. This script is a POSITIVE CONTROL:
it runs the EXACT same 3-arm M2b protocol (A original TR, B back-translation,
C English-direct) on a CONDITION WHERE CONTAMINATION IS PRESENT BY CONSTRUCTION,
and checks that the pre-registered M2b signature (B_acc < C_acc - 0.10) FIRES.

If the signature fires here but not on the real benchmarks, the null is
informative (probe works, benchmark is clean at this scale), not a probe failure.

DESIGN (planted memorization)
-----------------------------
For each seed-42 item we build a "leaked" variant: the correct answer letter and
text are planted verbatim into the prompt, mimicking an item the model has
memorized. We then run the same A/B/C arms. A memorized-surface model should:
  - score ~100% on the planted (M1-style) arm,
  - show B (back-translation) collapsing relative to C (English-direct) because
    the planted surface is Turkish-anchored.
This reproduces the contamination signature the probe is built to detect.

Same transport as the main audit (core.make_client -> opencode-go). No new deps.
"""

from __future__ import annotations
import argparse, json, os, random, re, sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from trmlu_audit import core

DEFAULT_MODEL = "deepseek-v4-flash"
SEED = 42
N = 30


def _chat(client, model, system, user, max_tokens=1600):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0)
    msg = r.choices[0].message
    return (msg.content or "").strip() or (getattr(msg, "reasoning_content", None) or "")[-200:]


def item_text(row: Dict) -> str:
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["secenekler"])))


def planted_text(row: Dict) -> str:
    """Leak the correct answer verbatim into the prompt (M1-style memorization)."""
    ti = int(row["cevap"])
    truth = row["secenekler"][ti]
    return (item_text(row) +
            f"\n\n[NOT: The correct answer is ({chr(65+ti)}) {truth}.]")


def tr_to_en(client, text: str, model: str) -> str:
    sys_ = ("You are a faithful professional Turkish->English translator. "
            "Translate the whole question AND every choice. Keep letter labels "
            "(A, B, C, ...) exactly. Preserve meaning. Output ONLY the translation.")
    return _chat(client, model, sys_, text)


def en_to_tr(client, en: str, model: str) -> str:
    sys_ = ("You are a faithful professional English->Turkish translator. "
            "Translate back to Turkish. Keep letter labels exactly. Output ONLY the translation.")
    return _chat(client, model, sys_, en)


def answer_item(client, text: str, lang: str, model: str) -> str:
    sys_ = "You are taking a standardized multiple-choice exam. " + lang
    return _chat(client, model, sys_, text + "\n\nCevap (single letter A/B/C/D/E):")


def parse_letter(ans: str) -> int:
    m = re.search(r"\b([A-E])\b", ans or "")
    return ord(m.group(1)) - 65 if m else -1


def load_sample(limit: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
    rng = random.Random(seed)
    return rng.sample(ds, min(limit, len(ds)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "positive_control_2026-08-25.jsonl"))
    args = ap.parse_args()
    model = args.model

    print(f"[i] Positive control: planted-contamination 3-arm run (model={model}, n={args.limit}, seed={SEED})")
    sample = load_sample(args.limit, SEED)
    client = core.make_client()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # resume: skip item indices already written to --out
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for l in f:
                try:
                    done.add(json.loads(l)["i"])
                except Exception:
                    pass
    print(f"[i] resuming: {len(done)} items already done, skipping them")

    for i, row in enumerate(sample):
        if i in done:
            continue
        truth = int(row["cevap"])
        a_txt = item_text(row)
        a_ans = answer_item(client, a_txt, "Answer in Turkish (question is Turkish). ", model)
        planted = planted_text(row)
        b_en = tr_to_en(client, planted, model)
        b_tr = en_to_tr(client, b_en, model)
        b_ans = answer_item(client, b_tr, "Answer in Turkish (question is Turkish). ", model)
        c_ans = answer_item(client, b_en, "Answer in English (item is English). ", model)

        rec = {
            "i": i, "truth": truth,
            "A_letter": parse_letter(a_ans), "A_correct": parse_letter(a_ans) == truth,
            "B_letter": parse_letter(b_ans), "B_correct": parse_letter(b_ans) == truth,
            "C_letter": parse_letter(c_ans), "C_correct": parse_letter(c_ans) == truth,
            "A_raw": a_ans[:60], "B_raw": b_ans[:60], "C_raw": c_ans[:60],
        }
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{i+1}/{len(sample)}] A={rec['A_correct']} B={rec['B_correct']} C={rec['C_correct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
