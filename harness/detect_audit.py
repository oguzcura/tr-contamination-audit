"""
Turkish cross-lingual contamination audit harness.
Probes TR-MMLU (alibayram/turkish_mmlu) for benchmark contamination using
API-only detection methods (Black Box + choice-substitution perturbation).

Run:
  uv run python detect_audit.py --split mmlu --limit 200 --models deepseek  gpt4o
Requires: HF token (gated dataset) + OPENROUTER_API_KEY in environment.
"""
from __future__ import annotations
import argparse, os, random, sys
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Dataset loading (TR-MMLU, gated). Requires HF_TOKEN.
# ---------------------------------------------------------------------------
def load_tr_mmlu(split: str = "mmlu", limit: int | None = None, seed: int = 42):
    """Load the gated official TR-MMLU split. Requires HF_TOKEN env var.
    Falls back to the official raw/JSON if gated auth is missing and dataset
    is already cached locally."""
    if not os.getenv("HF_TOKEN"):
        print("[!] HF_TOKEN not set. TR-MMLU is gated (CC BY-NC-ND 4.0).",
              file=sys.stderr)
    from datasets import load_dataset
    ds = load_dataset("alibayram/turkish_mmlu", split=split)
    # columns: bolum (subject), konu (topic), soru (question),
    #          cevap (answer id), aciklama (explanation), secenekler (choices)
    rows = list(ds)
    rng = random.Random(seed)
    if limit:
        rows = rng.sample(rows, min(limit, len(rows)))
    return rows


# ---------------------------------------------------------------------------
# 2. Model client (OpenRouter, DeepSeek v4 Flash-friendly).
# ---------------------------------------------------------------------------
def make_client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=os.getenv("OPENROUTER_API_KEY"))


def ask(client, model: str, question: str, choices: List[str]) -> Dict:
    """Ask the model to answer a multiple-choice question. Return raw response."""
    prompt = f"Soru (Turkish): {question}\n\nSecenekler:\n" + "\n".join(
        f"{chr(65+i)}) {c}" for i, c in enumerate(choices)) + "\n\nCevap (A/B/C/D):"
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600, temperature=0)  # reasoning models need budget to finish
    msg = r.choices[0].message
    content = msg.content
    # Reasoning models (e.g. DeepSeek v4 Flash) emit a hidden `reasoning`
    # chain and can leave `content` None. Fall back to the reasoning text.
    if content is None or not content.strip():
        reasoning = getattr(msg, "reasoning", None)
        if reasoning:
            content = "[" + str(reasoning)[:80] + "]"
        else:
            content = "<NO_ANSWER>"
    return {"model": model, "raw": content.strip()}


# ---------------------------------------------------------------------------
# 3. Choice-substitution perturbation probe (Yao 2406.13236v2, Deep-Contam).
#    Injects correct answers from *other* questions as distractor choices.
#    A contaminated model overfits to memorized choices.)
# ---------------------------------------------------------------------------
def build_perturbed(rows: List[Dict], idx: int) -> List[str]:
    """Replace one distractor choice with the correct answer of another question,
    to test whether the model is memorizing choices rather than reasoning."""
    correct = rows[idx]["secenekler"][int(rows[idx]["cevap"])]
    other_idx = (idx + 1) % len(rows)
    other_correct = rows[other_idx]["secenekler"][int(rows[other_idx]["cevap"])]
    perturbed = list(rows[idx]["secenekler"])
    # find a non-correct position and plant the foreign correct answer
    for i, c in enumerate(perturbed):
        if c != correct:
            perturbed[i] = other_correct
            break
    return perturbed


def run_blackbox(client, model: str, rows: List[Dict], limit: int) -> List[Dict]:
    """Black Box test: accuracy on the benchmark as a forward-pass baseline."""
    out = []
    for i, row in enumerate(rows[:limit]):
        out.append({"i": i, "subject": row["bolum"], "type": "blackbox",
                    **ask(client, model, row["soru"], row["secenekler"])})
    return out


def run_choice_substitution(client, model: str, rows: List[Dict], limit: int) -> List[Dict]:
    """Choice-substitution probe: does tightening the answer space change output?"""
    out = []
    for i in range(min(limit, len(rows))):
        choices = build_perturbed(rows, i)
        out.append({"i": i, "subject": rows[i]["bolum"], "type": "choice_sub",
                    **ask(client, model, rows[i]["soru"], choices)})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mmlu")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--probe", choices=["blackbox", "choice_sub", "both"], default="both")
    ap.add_argument("--models", nargs="+", default=["deepseek/deepseek-v4-flash-0731"])
    ap.add_argument("--out", default="results/pilot.jsonl")
    args = ap.parse_args()

    print(f"[i] Loading TR-MMLU split '{args.split}' ...")
    rows = load_tr_mmlu(args.split, limit=args.limit)
    print(f"[i] Loaded {len(rows)} questions (subjects: "
          f"{sorted({r['bolum'] for r in rows})[:8]} ...)")
    client = make_client()

    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for model in args.models:
            if args.probe in ("blackbox", "both"):
                for rec in run_blackbox(client, model, rows, args.limit):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.probe in ("choice_sub", "both"):
                for rec in run_choice_substitution(client, model, rows, args.limit):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[✓] Pilot results written to {args.out}")