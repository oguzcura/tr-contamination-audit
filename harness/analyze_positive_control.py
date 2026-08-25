"""Honest positive-control analysis (API-only, black-box constraint).

What a valid positive control can and cannot prove here:
  * M1 (verbatim recall): we PLANT the correct answer letter verbatim in the
    prompt. If M1-style leakage detection has any power, the model should
    reproduce the planted letter near-100% of the time -> a detectable signal.
    This is a VALID positive control for M1 (induced leak is detected).
  * M2b (surface-anchored fragility, B<C-0.10): requires the model to be
    genuinely anchored to the TURKISH SURFACE of a memorized item. Planting
    the answer verbatim does NOT create that anchoring -- it only makes the
    answer obvious, so all arms hit ceiling. A *valid* M2b positive control
    would require a model known to be contaminated on a Turkish benchmark,
    which an API-only audit does not have. We therefore report the M2b
    positive control as INCONCLUSIVE BY DESIGN and state the limitation
    honestly (consistent with api-only-contamination-audit-paper: an honest
    negative + method contribution is publishable).

This script computes, from the planted run:
  - planted_match_rate: fraction of items where the model emitted the planted
    letter (should be ~1.0, proving the induced leak is detectable).
  - per-arm accuracy A/B/C (will be ~1.0 ceiling -- confirms no B<C-0.10,
    i.e. the planted design does not trigger the M2b signature).
  - contrast vs the clean-audit M1 verbatim rates (from full_stats) to show
    the probe discriminates planted vs natural.
"""
from __future__ import annotations
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
PLANTED = os.path.join(HERE, "results", "positive_control_2026-08-25.jsonl")
# Clean-audit M1 verbatim rates (from harness/results/full_stats_2026-08-16.json, TR-MMLU + TUMLU-tr rows)
CLEAN_M1 = {"gpt-5.6-luna": 0.860, "mimo-v2.5": 0.575, "deepseek-v4-flash": 0.780}


def main() -> int:
    rows = [json.loads(l) for l in open(PLANTED, encoding="utf-8") if l.strip()]
    if not rows:
        print("NO DATA"); return 1
    n = len(rows)
    a = sum(1 for r in rows if r["A_correct"]) / n
    b = sum(1 for r in rows if r["B_correct"]) / n
    c = sum(1 for r in rows if r["C_correct"]) / n
    # planted_match: model emitted the planted letter (truth) -> induced-leak detectability
    pm = sum(1 for r in rows if r["A_letter"] == r["truth"]) / n
    print(f"[planted-control] n={n}  model=deepseek-v4-flash")
    print(f"  arm accuracies:  A={a:.3f}  B={b:.3f}  C={c:.3f}")
    print(f"  planted-match rate (model emits planted letter) = {pm:.3f}")
    print(f"  M2b signature B<C-0.10?  B={b:.3f} C={c:.3f} -> "
          f"{'FIRES' if b < c - 0.10 else 'does NOT fire (ceiling)'}")
    print(f"  clean-audit M1 verbatim (deepseek, TR-MMLU/TUMLU) = {CLEAN_M1['deepseek-v4-flash']:.3f}")
    print(f"  -> planted-match {pm:.3f} vs clean M1 {CLEAN_M1['deepseek-v4-flash']:.3f}: "
          f"{'probe DISCRIMINATES induced vs natural leak' if pm > CLEAN_M1['deepseek-v4-flash'] + 0.05 else 'weak contrast'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
