"""Analyze the Day-8 FULL run (multi-benchmark / multi-model, n=200).

Reads results/full_audit_2026-08-16.jsonl (plus spend checkpoints) and
computes, per benchmark x model cell (design paper1_contamination_audit.md §4
with the Day-8 approved amendments):

  - M1:  verbatim-hit rate + Wilson 95% CI; mean 8-gram overlap (+/- sd)
  - M2a: flip-follow rate + Wilson 95% CI; paired McNemar exact vs M2b-A
  - M2b: per-arm accuracy (A/B/C) + Wilson CI on the fully-matched denominator;
         McNemar exact (B-vs-A, C-vs-A, C-vs-B)
  - Bonferroni within benchmark x probe family (raw + corrected p)
  - Consensus TWICE (fix 1, pre-registered):
      * RAW (§0): >=2/3 modalities indicate AND >=1 dynamic arm p<0.05
        after Bonferroni, where "indicate" is the pilot operationalization
        (M1: mean_overlap8>0.5 AND verbatim_rate>0.5; M2a: flip_follow_rate
        >0.5; M2b: B_acc < C_acc - 0.10).
      * M1-DISCOUNT: same rule with M1 STRUCK from the modality count
        (M1 is instruction-following-confounded by design; the flagged pilot
        cells all relied on M1 as their second modality - Day-8 fix 1).
    Both readings are reported for every cell; the adjudication table shows
    the delta.
  - QA cells (RAGTurk) use the fix-3 robust matcher (answer_correct), with a
    lenient-matcher sensitivity reading per arm.
  - Spend: per-model tokens/cost + checkpoint log + run events (detects
    resumption/interruption for the report).

Output: results/full_stats_2026-08-16.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from math import sqrt

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results", "full_audit_2026-08-16.jsonl")
SPEND_IN = os.path.join(HERE, "results", "spend_full_2026-08-16.jsonl")
LABELED = os.path.join(HERE, "results", "ragturk_matcher_labeled.json")
OUT_JSON = os.path.join(HERE, "results", "full_stats_2026-08-16.json")

BENCHMARKS = ["tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"]
MODELS = ["gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"]
MC = ("tr_mmlu", "tumlu_tr")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(c - h, 0.0), 4), round(min(c + h, 1.0), 4))


def mcnemar_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(a, b), n, p=0.5).pvalue)


def mean_std(vals):
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return (round(m, 4), round(sd, 4))


def load():
    recs, events, n_skipped = [], [], 0
    with open(IN, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                n_skipped += 1
                continue
            (events if r.get("event") else recs).append(r)
    return recs, events, n_skipped


def group(recs):
    g = defaultdict(dict)
    for r in recs:
        g[(r["benchmark"], r["model"], r["probe"])][r["item_idx"]] = r
    return g


def cell_m1(grp, bench, model):
    recs = grp.get((bench, model, "M1"), {})
    ok = [r for r in recs.values() if r.get("ok")]
    n = len(ok)
    hits = sum(1 for r in ok if r.get("verbatim_hit"))
    ovs = [r.get("overlap8", 0.0) for r in ok]
    m, sd = mean_std(ovs)
    lo, hi = wilson(hits, n)
    return {"n": n, "verbatim_hits": hits,
            "verbatim_rate": round(hits / n, 4) if n else None,
            "verbatim_ci95": [lo, hi], "mean_overlap8": m, "overlap8_sd": sd}


def cell_m2a(grp, bench, model):
    m2a = grp.get((bench, model, "M2a"), {})
    base = grp.get((bench, model, "M2b_A"), {})
    ok = [r for r in m2a.values() if r.get("ok")]
    flips = [r for r in ok if r.get("flip_follow")]
    n_flip = len(ok)
    lo, hi = wilson(len(flips), n_flip)
    pos = neg = 0
    for item, r in m2a.items():
        b = base.get(item)
        if b is None or not b.get("ok") or not r.get("ok"):
            continue
        if b.get("answer_correct") and not r.get("answer_correct"):
            pos += 1
        elif (not b.get("answer_correct")) and r.get("answer_correct"):
            neg += 1
    n_pair = pos + neg
    return {"n_flip": n_flip, "flip_follows": len(flips),
            "flip_follow_rate": round(len(flips) / n_flip, 4) if n_flip else None,
            "flip_follow_ci95": [lo, hi],
            "mcnemar_discordant": [pos, neg], "mcnemar_n_pairs": n_pair,
            "mcnemar_p_raw": round(mcnemar_p(pos, neg), 4)}


def cell_m2b(grp, bench, model):
    arms = {"A": "M2b_A", "B": "M2b_B", "C": "M2b_C"}
    recs = {a: grp.get((bench, model, p), {}) for a, p in arms.items()}

    def parsed(r):
        if not r or not r.get("ok"):
            return False
        if not (r.get("raw") or r.get("content") or "").strip():
            return False
        if bench in MC:
            return r.get("parsed_letter") is not None
        return True

    items = sorted(set.intersection(*[set(recs[a].keys()) for a in arms]))
    matched = [i for i in items if all(parsed(recs[a].get(i)) for a in arms)]

    def acc(a):
        k = sum(1 for i in matched if recs[a][i].get("answer_correct"))
        lo, hi = wilson(k, len(matched))
        return {"correct": k, "n": len(matched),
                "acc": round(k / len(matched), 4) if matched else None,
                "ci95": [lo, hi]}

    out = {"matched_n": len(matched), "arms": {a: acc(a) for a in arms},
           "mcnemar": {}}
    a_ok = lambda i, a: bool(recs[a][i].get("answer_correct"))
    for test in ("B", "C"):
        pos = sum(1 for i in matched if a_ok(i, "A") and not a_ok(i, test))
        neg = sum(1 for i in matched if not a_ok(i, "A") and a_ok(i, test))
        out["mcnemar"][f"{test}_vs_A"] = {"discordant": [pos, neg],
                                          "p_raw": round(mcnemar_p(pos, neg), 4)}
    pos = sum(1 for i in matched if a_ok(i, "B") and not a_ok(i, "C"))
    neg = sum(1 for i in matched if not a_ok(i, "B") and a_ok(i, "C"))
    out["mcnemar"]["C_vs_B"] = {"discordant": [pos, neg],
                                "p_raw": round(mcnemar_p(pos, neg), 4)}
    # lenient-matcher sensitivity (RAGTurk QA cells; fix-3 disclosure)
    if bench == "ragturk_formal5k":
        out["lenient_sensitivity"] = {}
        for a in arms:
            k = sum(1 for i in matched if recs[a][i].get("qa_match_lenient"))
            out["lenient_sensitivity"][a] = {"correct": k, "n": len(matched),
                                             "acc": round(k / len(matched), 4) if matched else None}
    return out


def collect_pvals(cells):
    fam = defaultdict(list)
    for (bench, model), c in cells.items():
        for probe, info in c.items():
            if probe == "M2a":
                fam[(bench, "M2a")].append((model, "M2a_mcnemar", info["mcnemar_p_raw"]))
            elif probe == "M2b":
                for k, v in info["mcnemar"].items():
                    fam[(bench, "M2b")].append((model, k, v["p_raw"]))
    return fam


def consensus(c, bench, discount_m1=False):
    """Modality indications; discount_m1 -> M1 struck (fix 1 sensitivity)."""
    m1 = c.get("M1", {})
    m2a = c.get("M2a", {})
    m2b = c.get("M2b", {})
    ind = {}
    ind["M1"] = (not discount_m1) and bool(
        m1.get("mean_overlap8") is not None and m1["mean_overlap8"] > 0.5
        and (m1.get("verbatim_rate") or 0) > 0.5)
    ind["M2a"] = bool(bench != "ragturk_formal5k"
                      and (m2a.get("flip_follow_rate") or 0) > 0.5)
    b_a, c_a = (m2b.get("arms", {}).get("B", {}).get("acc"),
                m2b.get("arms", {}).get("C", {}).get("acc"))
    ind["M2b"] = bool(b_a is not None and c_a is not None and b_a < c_a - 0.10)
    return ind


def main():
    recs, events, n_skipped = load()
    grp = group(recs)
    cells = {}
    for bench in BENCHMARKS:
        for model in MODELS:
            c = {"M1": cell_m1(grp, bench, model)}
            if bench != "ragturk_formal5k":
                c["M2a"] = cell_m2a(grp, bench, model)
            c["M2b"] = cell_m2b(grp, bench, model)
            cells[(bench, model)] = c

    fam = collect_pvals(cells)
    adj = {}
    for (bench, probe), tests in fam.items():
        k = len(tests)
        for model, name, p in tests:
            adj[(bench, model, probe, name)] = round(min(1.0, p * k), 4)

    starts = [e for e in events if e.get("event") == "run_start"]
    ends = [e for e in events if e.get("event") == "run_end"]
    out = {
        "phase": "full_2026-08-16", "n_per_benchmark": 200, "seed": 42,
        "run_events": {"run_start": len(starts), "run_end": len(ends),
                       "resumed_or_interrupted": len(starts) > 1,
                       "starts": starts, "ends": ends},
        "rule": ("contaminated iff >=2/3 modalities indicate AND >=1 dynamic "
                 "arm (M2a/M2b) p<0.05 after Bonferroni; RAW + M1-DISCOUNT "
                 "readings both reported (Day-8 fix 1)"),
        "bonferroni_family_sizes": {f"{b}|{p}": len(v) for (b, p), v in fam.items()},
        "cells": {}, "spend": {}, "matcher_doc": {}}

    for (bench, model), c in cells.items():
        c_out = {}
        for probe in ("M1", "M2a", "M2b"):
            if probe not in c:
                continue
            c_out[probe] = dict(c[probe])
        if "M2a" in c_out:
            key = (bench, model, "M2a", "M2a_mcnemar")
            c_out["M2a"]["mcnemar_p_bonf"] = adj.get(key, 1.0)
        if "M2b" in c_out:
            for name, v in c_out["M2b"]["mcnemar"].items():
                v["p_bonf"] = adj.get((bench, model, "M2b", name), 1.0)
        for label, discount in (("raw", False), ("m1_discount", True)):
            ind = consensus(c, bench, discount_m1=discount)
            dynamic = []
            if bench != "ragturk_formal5k":
                dynamic.append(c_out.get("M2a", {}).get("mcnemar_p_bonf", 1.0))
            dynamic += [v["p_bonf"] for v in c_out.get("M2b", {}).get("mcnemar", {}).values()]
            dyn_min = min(dynamic) if dynamic else 1.0
            n_ind = sum(1 for v in ind.values() if v)
            flagged = n_ind >= 2 and dyn_min < 0.05
            c_out[label] = {
                "modality_indications": ind,
                "n_modalities_indicate": n_ind,
                "min_dynamic_arm_p_bonf": dyn_min,
                "consensus_call": ("CONTAMINATED (pre-registered rule)" if flagged
                                   else "no significant contamination detected")}
        out["cells"][f"{bench}|{model}"] = c_out

    # ---- spend (every logged call incl. translations + events excluded) ----
    sp = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0,
                              "retry_hard": 0, "prompt_tokens": 0, "cached_tokens": 0,
                              "completion_tokens": 0, "reasoning_tokens": 0,
                              "cost_est_usd": 0.0, "cost_est_n": 0})
    for r in recs:
        s = sp[r["model"]]
        s["calls"] += 1
        s["ok"] += int(r.get("ok"))
        s["fail"] += int(not r.get("ok"))
        s["retries"] += int(r.get("retries") or 0)
        s["retry_hard"] += int(bool(r.get("retry_hard")))
        for tk in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
            s[tk] += int(r.get(tk) or 0)
        if r.get("cost_est_usd") is not None:
            s["cost_est_usd"] += r["cost_est_usd"]
            s["cost_est_n"] += 1
    out["spend"] = {k: {**v, "cost_est_usd": round(v["cost_est_usd"], 6),
                        "has_rate": v["cost_est_n"] > 0} for k, v in sp.items()}
    out["spend"]["TOTAL_calls"] = len(recs)
    out["spend"]["TOTAL_tokens"] = sum(r.get("total_tokens") or 0 for r in recs)
    out["spend"]["TOTAL_retries"] = sum(int(r.get("retries") or 0) for r in recs)
    total_cost = sum(v["cost_est_usd"] for k, v in out["spend"].items()
                     if k.startswith(("gpt", "deepseek", "mimo")) and v["has_rate"])
    out["spend"]["TOTAL_cost_est_usd_rated"] = round(total_cost, 6)
    chk = []
    if os.path.exists(SPEND_IN):
        with open(SPEND_IN, encoding="utf-8") as fh:
            chk = [json.loads(l) for l in fh if l.strip()]
    out["spend"]["checkpoints"] = chk

    # matcher doc (fix 3; measured pre-run on the labeled pilot set)
    if os.path.exists(LABELED):
        lab = json.load(open(LABELED, encoding="utf-8"))
        out["matcher_doc"] = {"n_labeled_pairs": lab["n"],
                              "confusion": lab["confusion"],
                              "artifact": "results/ragturk_matcher_labeled.json"}

    if n_skipped:
        out["unparseable_lines"] = n_skipped
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_JSON}")
    for (bench, model), c in sorted(cells.items()):
        cell = out["cells"][f"{bench}|{model}"]
        print(f"{bench:16s} {model:18s} RAW={cell['raw']['consensus_call'][:32]:32s} "
              f"DISCOUNT={cell['m1_discount']['consensus_call'][:32]:32s} "
              f"(n_mod RAW={cell['raw']['n_modalities_indicate']} "
              f"DISCOUNT={cell['m1_discount']['n_modalities_indicate']}, "
              f"dyn_p_bonf={cell['raw']['min_dynamic_arm_p_bonf']:.4f})")


if __name__ == "__main__":
    main()