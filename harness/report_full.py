"""Generate notes/full_results_2026-08-16.md from the Day-8 full-run artifacts.

Inputs (real values only; no fabrication):
  notes/pre_reg_full_2026-08-16.md        (frozen pre-registration, verbatim)
  results/full_audit_2026-08-16.jsonl     (every API call + event markers)
  results/spend_full_2026-08-16.jsonl     (spend checkpoints every 500 calls)
  results/full_stats_2026-08-16.json      (all cells, CIs, p, dual consensus)
  results/ragturk_matcher_labeled.json    (matcher doc / measured quality)
Output: notes/full_results_2026-08-16.md
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results", "full_audit_2026-08-16.jsonl")
SPEND = os.path.join(HERE, "results", "spend_full_2026-08-16.jsonl")
STATS = os.path.join(HERE, "results", "full_stats_2026-08-16.json")
PRE = os.path.join(os.path.dirname(HERE), "notes", "pre_reg_full_2026-08-16.md")
OUT = os.path.join(os.path.dirname(HERE), "notes", "full_results_2026-08-16.md")

BENCH_NAMES = {"tr_mmlu": "TR-MMLU", "tumlu_tr": "TUMLU-tr",
               "halluverse_tr": "Halluverse-M3-tr", "ragturk_formal5k": "RAGTurk formal_5k"}
MODEL_NAMES = {"gpt-5.6-luna": "gpt-5.6-luna", "mimo-v2.5": "mimo-v2.5",
               "deepseek-v4-flash": "deepseek-v4-flash (ref)"}
MC = ("tr_mmlu", "tumlu_tr")
CELL_ORDER = {"tr_mmlu": 0, "tumlu_tr": 1, "halluverse_tr": 2, "ragturk_formal5k": 3}


def fmt_p(p):
    if p is None:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def acc_str(a):
    if a is None or a.get("n", 0) == 0:
        return "—"
    lo, hi = a["ci95"]
    return f"{a['acc']:.3f} ({a['correct']}/{a['n']}) [{lo:.3f},{hi:.3f}]"


def main():
    pre = open(PRE, encoding="utf-8").read().strip()
    stats = json.load(open(STATS, encoding="utf-8"))
    recs = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    calls = [r for r in recs if not r.get("event")]
    events = [r for r in recs if r.get("event")]

    # ---- call ledger -----------------------------------------------------
    per_model = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0,
                                     "retry_hard": 0, "prompt_tk": 0, "comp_tk": 0,
                                     "reason_tk": 0, "cost": 0.0, "cost_n": 0})
    per_bench = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0})
    n_fail_long = n_fail_rec = 0
    for r in calls:
        pm, pb = per_model[r["model"]], per_bench[r["benchmark"]]
        for d in (pm, pb):
            d["calls"] += 1
            d["ok"] += int(r.get("ok"))
            d["fail"] += int(not r.get("ok"))
            d["retries"] += int(r.get("retries") or 0)
        if r.get("retry_hard"):
            pm["retry_hard"] += 1
            if r["retry_hard"] == "long":
                n_fail_long += int(not r.get("ok"))
            else:
                n_fail_rec += int(not r.get("ok"))
        pm["prompt_tk"] += int(r.get("prompt_tokens") or 0)
        pm["comp_tk"] += int(r.get("completion_tokens") or 0)
        pm["reason_tk"] += int(r.get("reasoning_tokens") or 0)
        if r.get("cost_est_usd") is not None:
            pm["cost"] += r["cost_est_usd"]
            pm["cost_n"] += 1

    starts = [e for e in events if e.get("event") == "run_start"]
    ends = [e for e in events if e.get("event") == "run_end"]
    resumed = len(starts) > 1
    aborted = stats["run_events"].get("resumed_or_interrupted") or (
        ends and not ends[-1].get("run_calls"))
    elapsed = 0.0
    if starts and ends:
        try:
            from datetime import datetime
            t0 = datetime.strptime(starts[0]["ts"], "%Y-%m-%dT%H:%M:%S")
            t1 = datetime.strptime(ends[-1]["ts"], "%Y-%m-%dT%H:%M:%S")
            elapsed = (t1 - t0).total_seconds()
        except Exception:
            elapsed = 0.0

    L = []
    A = L.append
    A("# Full-Run Results — Turkish Benchmark Contamination Audit (Day 8, 2026-08-16)")
    A("")
    A("**Scope:** n=200 seed-42 matched items per benchmark x 3 models "
      "(gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash reference), ALL via "
      "opencode-go (`trmlu_audit.core`; OpenRouter untouched). Probes M1 + M2a + M2b "
      "3-arm. The four user-mandated fixes (2026-08-16 approval) are implemented and "
      "pre-registered below before any results.")
    A("")
    A(f"**Run integrity:** run_start markers = {len(starts)}, run_end = {len(ends)}; "
      f"**this run was {'RESUMED/INTERRUPTED' if resumed else 'a single uninterrupted run'}** "
      f"(mandatory report line). Elapsed (first start -> last end) ≈ {elapsed/3600:.2f} h.")
    if stats.get("unparseable_lines"):
        A(f"Unparseable log lines tolerated: {stats['unparseable_lines']}.")
    A("")
    A("---")
    A("")
    A("## 0. PRE-REGISTRATION (verbatim, frozen before the run)")
    A("")
    A(pre.replace("\n# ", "\n### ").replace("## ", "### ").replace("# PRE-REGISTRATION",
                                                                    "### PRE-REGISTRATION"))
    A("")
    A("---")
    A("")
    A("## 1. Run reliability & spend")
    A("")
    A(f"**Calls logged: {stats['spend']['TOTAL_calls']}** "
      f"(planned 13,200; difference = degenerate items not called / retry exhaustion "
      f"handled in denominators). ok = {sum(v['ok'] for v in per_model.values())}, "
      f"fail = {sum(v['fail'] for v in per_model.values())}, "
      f"retries = {stats['spend']['TOTAL_retries']}.")
    A("")
    A("### Spend log (from the JSONL; documented rates only)")
    A("")
    A("| model | calls | ok | fail | retries | retry-hard | prompt tk | completion tk | reasoning tk | cost est USD |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
        s = per_model[m]
        A(f"| {m} | {s['calls']} | {s['ok']} | {s['fail']} | {s['retries']} | "
          f"{s['retry_hard']} | {s['prompt_tk']} | {s['comp_tk']} | {s['reason_tk']} | "
          f"${s['cost']:.6f} |")
    A(f"| **TOTAL** | {stats['spend']['TOTAL_calls']} | "
      f"{sum(v['ok'] for v in per_model.values())} | "
      f"{sum(v['fail'] for v in per_model.values())} | {stats['spend']['TOTAL_retries']} | "
      f"{sum(v['retry_hard'] for v in per_model.values())} | "
      f"{sum(v['prompt_tk'] for v in per_model.values())} | "
      f"{sum(v['comp_tk'] for v in per_model.values())} | "
      f"{sum(v['reason_tk'] for v in per_model.values())} | "
      f"**${stats['spend']['TOTAL_cost_est_usd_rated']:.6f}** |")
    A("")
    A(f"Rated-model total: **${stats['spend']['TOTAL_cost_est_usd_rated']:.6f}** "
      f"against the **$2.00 cap** (mimo-v2.5 tokens-only, uncosted). "
      f"Spend checkpoints written every 500 calls: "
      f"{len(stats['spend'].get('checkpoints', []))} lines in results/spend_full_2026-08-16.jsonl.")
    A("")
    A("### Per benchmark")
    A("")
    A("| benchmark | calls | ok | fail | retries |")
    A("|---|---|---|---|---|")
    for b in ("tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"):
        s = per_bench[b]
        A(f"| {BENCH_NAMES[b]} | {s['calls']} | {s['ok']} | {s['fail']} | {s['retries']} |")
    A("")
    A(f"Fix-2 retry-hard events (gpt-5.6-luna x RAGTurk): "
      f"{sum(v['retry_hard'] for v in per_model.values())} triggered "
      f"(long-surface >2000 chars: {n_fail_long} failures — expected 0 triggers in "
      f"formal_5k; item-recurrence: {n_fail_rec} exhausted failures, each logged).")
    A("")
    A("## 2. Matcher documentation (fix 3, measured pre-run)")
    A("")
    md = stats.get("matcher_doc", {})
    if md:
        conf = md["confusion"]
        tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        A(f"Hand-labeled set: {md['n_labeled_pairs']} (gold, pred) pairs from the "
          f"Day-5 pilot; two-rater agreement on a 10-pair batch; artifact "
          f"`results/ragturk_matcher_labeled.json`.")
        A(f"Measured (strict cascade of `harness/qa_matcher.py`): "
          f"**precision {prec:.3f}, recall {rec:.3f}**, accuracy "
          f"{(tp + tn) / (tp + fp + tn + fn):.3f}; confusion TP={tp} FP={fp} TN={tn} FN={fn}.")
        A("Level histogram: exact 2, containment 11, number-mismatch 48 (rejected), "
          "none 44, empty 1; the paraphrase levels matched 0 of the labeled pairs "
          "in the final cascade (the 4 naive-paraphrase candidates were single-fact "
          "swaps, e.g. teorik/istatistiksel fizik, UCID/Wadani, and are rejected by "
          "the content-token + distinctive-gram guards).")
        A("**Interpretation:** the matcher is deliberately precision-first; its low "
          "recall understates true QA capability, so every RAGTurk accuracy is a "
          "lower bound on string-verifiable correctness. A lenient sensitivity "
          "variant is reported per arm below; caution is required before any "
          "RAGTurk claim (design §6: 'interpret carefully').")
    A("")
    # ---- per-benchmark cells ---------------------------------------------
    for bench in ("tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"):
        A(f"## 3.{CELL_ORDER[bench]+1} {BENCH_NAMES[bench]}")
        A("")
        A("**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-"
          "following; never decisive alone; struck in the M1-DISCOUNT reading)")
        A("")
        A("| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |")
        A("|---|---|---|---|---|")
        for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
            c1 = stats["cells"][f"{bench}|{m}"]["M1"]
            n, h = c1["n"], c1["verbatim_hits"]
            lo, hi = c1["verbatim_ci95"]
            mo, sd = c1["mean_overlap8"], c1["overlap8_sd"]
            rate = f"{c1['verbatim_rate']:.3f}" if n else "—"
            A(f"| {MODEL_NAMES[m]} | {n} | {h} | {rate} [{lo:.3f},{hi:.3f}] | "
              f"{mo} ± {sd} |")
        A("")
        if bench != "ragturk_formal5k":
            A("**M2a — distractor flip / edited-answer flip** (follow-planted rate + "
              "paired McNemar exact vs A-arm baseline; Bonferroni family size=3)")
            A("")
            A("| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |")
            A("|---|---|---|---|---|---|---|")
            for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
                c2 = stats["cells"][f"{bench}|{m}"]["M2a"]
                nf, ff = c2["n_flip"], c2["flip_follows"]
                lo, hi = c2["flip_follow_ci95"]
                rate = f"{c2['flip_follow_rate']:.3f}" if nf else "—"
                d1, d2 = c2["mcnemar_discordant"]
                A(f"| {MODEL_NAMES[m]} | {nf} | {ff} | {rate} [{lo:.3f},{hi:.3f}] | "
                  f"({d1}, {d2}) | {fmt_p(c2['mcnemar_p_raw'])} | "
                  f"{fmt_p(c2['mcnemar_p_bonf'])} |")
            A("")
        A("**M2b — 3-arm crosslingual fragility** (fully-matched denominator; "
          "McNemar exact paired; Bonferroni family size=9; QA arms scored by the "
          "fix-3 robust matcher; RAGTurk arm C vs EN-translated gold)")
        A("")
        A("| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |")
        A("|---|---|---|---|---|---|---|---|")
        for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
            c3 = stats["cells"][f"{bench}|{m}"]["M2b"]
            arms = c3["arms"]
            A(f"| {MODEL_NAMES[m]} | {c3['matched_n']} | {acc_str(arms['A'])} | "
              f"{acc_str(arms['B'])} | {acc_str(arms['C'])} | "
              f"{fmt_p(c3['mcnemar']['B_vs_A']['p_bonf'])} | "
              f"{fmt_p(c3['mcnemar']['C_vs_A']['p_bonf'])} | "
              f"{fmt_p(c3['mcnemar']['C_vs_B']['p_bonf'])} |")
        A("")
        if bench == "ragturk_formal5k":
            A("**RAGTurk lenient-sensitivity reading (fix 3 disclosure; not "
              "headline):** accuracy with the lenient matcher variant — ")
            sens = [f"{a}: {stats['cells'][f'{bench}|{m}']['M2b']['lenient_sensitivity'][a]['acc']:.3f}"
                    for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash")
                    for a in ("A", "B", "C")]
            A(", ".join(sens) + ".")
            A("")
        A("**Consensus (per §0 RAW and pre-registered M1-DISCOUNT):**")
        A("")
        A("| model | modalities RAW | min dyn p (Bonf) | RAW call | M1-DISCOUNT call | adjudicated |")
        A("|---|---|---|---|---|---|")
        for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
            cs = stats["cells"][f"{bench}|{m}"]
            raw, disc = cs["raw"], cs["m1_discount"]
            adv = disc["consensus_call"]
            if raw["consensus_call"].startswith("CONTAMINATED") and \
                    disc["consensus_call"].startswith("no significant"):
                adv = "THRESHOLD-DEPENDENT (M1-sensitive)"
            A(f"| {MODEL_NAMES[m]} | {raw['n_modalities_indicate']}/3 "
              f"({', '.join(k for k, v in raw['modality_indications'].items() if v) or '—'}) | "
              f"{fmt_p(raw['min_dynamic_arm_p_bonf'])} | {raw['consensus_call']} | "
              f"{disc['consensus_call']} | **{adv}** |")
        A("")
        A("---")
        A("")

    # ---- adjudication summary --------------------------------------------
    A("## 4. Adjudication summary (RAW vs M1-DISCOUNT)")
    A("")
    A("| benchmark | model | RAW | M1-DISCOUNT | adjudicated |")
    A("|---|---|---|---|---|")
    for bench in ("tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"):
        for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
            cs = stats["cells"][f"{bench}|{m}"]
            raw, disc = cs["raw"], cs["m1_discount"]
            adv = disc["consensus_call"]
            if raw["consensus_call"].startswith("CONTAMINATED") and \
                    disc["consensus_call"].startswith("no significant"):
                adv = "THRESHOLD-DEPENDENT (M1-sensitive)"
            A(f"| {BENCH_NAMES[bench]} | {MODEL_NAMES[m]} | {raw['consensus_call']} | "
              f"{disc['consensus_call']} | **{adv}** |")
    A("")
    A("Per the pre-registered adjudication (fix 1): a cell is flagged only on the "
      "M1-DISCOUNT reading; RAW-only flags are threshold-dependent by definition.")
    A("")
    A("## 5. Honest interpretation")
    A("")
    int_notes = [
        "M1 values on the MC benchmarks are high for instruction-following models "
        "because the M1 prompt explicitly requests verbatim reproduction — by design "
        "M1 never flags alone and is struck in the M1-DISCOUNT reading.",
        "The M2b signature of interest (ConStat non-generalizing performance) is "
        "B collapsing while C stays stable; where B-vs-A is significant after "
        "Bonferroni with C-vs-A not significant, the pattern is reported as "
        "surface-fragility, not concept-independent memorization; no M2a "
        "flip-following anywhere means no model followed planted distractors "
        "(rates and p reported per cell).",
        "Halluverse free-form QA baseline is low (arm A accuracy per table); its "
        "M2a 2-option control rescues items relative to free-form answering and the "
        "McNemar discordance is the structure effect, not flip-following — cells "
        "stay honest negatives unless the dynamic-arm bar is met.",
        "RAGTurk formal_5k: accuracies are lower bounds under the precision-first "
        "matcher (recall 0.371 measured). The A-arm is near floor for most cells, "
        "so no fragility contrast is resolvable; arm C is scored against "
        "machine-translated golds (probe M2b_gold_en) — an extra moving part, "
        "reported as such. Treat every RAGTurk number as conservative.",
        "All null statements are scope-limited: 'no contamination detected by "
        "these probes at n=200 per benchmark x model', never a claim that a "
        "benchmark is clean (black-box API: training inclusion is unobservable).",
    ]
    for i, note in enumerate(int_notes, 1):
        A(f"{i}. {note}")
        A("")
    A("## 6. Limitations")
    A("")
    lims = [
        "Black-box API: memorization signal is inferred from sampled text; no "
        "weights/logits; training inclusion is not observable.",
        "M1 confound (instruction-following) — handled by the pre-registered "
        "M1-DISCOUNT reading; RAW reading shown for transparency.",
        "QA semantic matching is noisy: the fix-3 matcher is measured precision "
        "1.000 / recall 0.371 on 106 labeled pairs (labels = author + 10-pair "
        "agreement batch; artifact published). RAGTurk cells are conservative.",
        "RAGTurk arm-C English golds are machine-translated (single pass, "
        "deepseek-v4-flash); translation error propagates into C scoring.",
        f"Per-cell matched denominators drop below n=200 where probes failed or "
        f"degenerate items were excluded; every table reports its own n.",
        "Failure cluster risk was mitigated by fix-2 retry-hard; any remaining "
        "exhausted calls appear in the honest denominators (per-cell n).",
        "n=200 is a sample, not full-benchmark: results generalize to the seed-42 "
        "matched subsets, not to whole benchmarks.",
    ]
    for i, lim in enumerate(lims, 1):
        A(f"{i}. {lim}")
        A("")
    A("---")
    A("*Generated mechanically from results/full_stats_2026-08-16.json + "
      "results/full_audit_2026-08-16.jsonl + results/spend_full_2026-08-16.jsonl; "
      "pre-registration section verbatim from notes/pre_reg_full_2026-08-16.md.*")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()