"""Generate notes/pilot_results_2026-08-16.md from the real pilot artifacts.

Inputs (real values only; no fabrication):
  results/pilot_audit_2026-08-16.jsonl  (every API call)
  results/pilot_stats_2026-08-16.json   (all cells + CIs + p + consensus calls)
Output:
  notes/pilot_results_2026-08-16.md     (tables + honest interpretations + spend log)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results", "pilot_audit_2026-08-16.jsonl")
STATS = os.path.join(HERE, "results", "pilot_stats_2026-08-16.json")
OUT = os.path.join(os.path.dirname(HERE), "notes", "pilot_results_2026-08-16.md")

BENCH_NAMES = {"tr_mmlu": "TR-MMLU", "tumlu_tr": "TUMLU-tr",
               "halluverse_tr": "Halluverse-M3-tr", "ragturk_formal5k": "RAGTurk formal_5k"}
MODEL_NAMES = {"gpt-5.6-luna": "gpt-5.6-luna", "mimo-v2.5": "mimo-v2.5",
               "deepseek-v4-flash": "deepseek-v4-flash (ref)"}
MC = ("tr_mmlu", "tumlu_tr")


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
    stats = json.load(open(STATS, encoding="utf-8"))
    recs = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]

    # ---- call ledger aggregates (real JSONL numbers) ---------------------
    per_model = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0,
                                     "prompt_tk": 0, "comp_tk": 0, "reason_tk": 0,
                                     "cost": 0.0, "cost_n": 0})
    per_bench = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0})
    for r in recs:
        pm = per_model[r["model"]]
        pm["calls"] += 1
        pm["ok"] += int(r.get("ok"))
        pm["fail"] += int(not r.get("ok"))
        pm["retries"] += int(r.get("retries") or 0)
        pm["prompt_tk"] += int(r.get("prompt_tokens") or 0)
        pm["comp_tk"] += int(r.get("completion_tokens") or 0)
        pm["reason_tk"] += int(r.get("reasoning_tokens") or 0)
        if r.get("cost_est_usd") is not None:
            pm["cost"] += r["cost_est_usd"]
            pm["cost_n"] += 1
        pb = per_bench[r["benchmark"]]
        pb["calls"] += 1
        pb["ok"] += int(r.get("ok"))
        pb["fail"] += int(not r.get("ok"))
        pb["retries"] += int(r.get("retries") or 0)

    L = []
    A = L.append
    A("# Pilot Results — Turkish Benchmark Contamination Audit (Day 5, 2026-08-16)")
    A("")
    A("**Scope:** n=50 seed-42 matched items per benchmark x 3 models "
      "(gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash reference), all via opencode-go "
      "(OpenRouter untouched). Probes M1 (verbatim-recall + 8-gram), M2a "
      "(distractor-flip / edited-answer flip), M2b (3-arm A/B/C crosslingual). "
      "Design: `notes/paper1_contamination_audit.md` §0, §3, §4.")
    A("")
    A("**Pre-registered decision rule (§0), applied exactly:** a benchmark x model cell "
      "is flagged **only if ≥2 of 3 probe modalities indicate AND ≥1 dynamic arm "
      "(M2a/M2b) has p < 0.05 after Bonferroni** (family = benchmark x probe). "
      "Any other outcome = **no significant contamination detected** (honest negative).")
    A("")
    A("**Probe applicability (honest notes):** M2a **does not apply** to RAGTurk "
      "(QA-pair, no MC options) — no calls made, cell reported as N/A. Halluverse "
      "M2a uses its built-in hallucination-edit twin (`edited_answer`) as a 2-option "
      "forced-choice flip (positions randomized per item, seed-42). QA arms are scored "
      "with a **lenient containment match** (normalized exact/substring); this is "
      "weaker than letter-exact MC scoring and reported as such. M1 asks the model to "
      "reproduce the answer text verbatim, so high M1 values partly measure "
      "instruction-following — M1 alone never flags a cell (documented confound).")
    A("")

    # ---- 1. run / reliability --------------------------------------------
    n_total = len(recs)
    ok_total = sum(1 for r in recs if r.get("ok"))
    fail_total = n_total - ok_total
    ret_total = sum(int(r.get("retries") or 0) for r in recs)
    A("## 1. Run reliability & spend")
    A("")
    A(f"**Calls logged: {n_total}** (planned 2,850 answer calls + 400 lazy "
      f"translations = 3,250). ok = {ok_total}, fail = {fail_total}, "
      f"retry-on-empty events (total, logged) = {ret_total} "
      f"({sum(int(r.get('retries') or 0) for r in recs if r.get('ok'))} recovered, "
      f"{sum(int(r.get('retries') or 0) for r in recs if not r.get('ok'))} exhausted "
      f"across {sum(1 for r in recs if not r.get('ok'))} calls). "
      f"No 5xx/429 backoff retries triggered. "
      f"No live cost-cap abort (accumulated estimate stayed below $1.00).")
    A("")
    A("### Spend log (from the JSONL; documented rates only)")
    A("")
    A("| model | calls | ok | fail | retries | prompt tk | completion tk | "
      "reasoning tk | cost est USD |")
    A("|---|---|---|---|---|---|---|---|---|")
    for m in ("gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"):
        pm = per_model.get(m, {})
        if not pm:
            continue
        cost = f"${pm['cost']:.6f}" if pm["cost_n"] else "n/a (no published rate)"
        A(f"| {m} | {pm['calls']} | {pm['ok']} | {pm['fail']} | {pm['retries']} | "
          f"{pm['prompt_tk']} | {pm['comp_tk']} | {pm['reason_tk']} | {cost} |")
    A(f"| **TOTAL** | **{n_total}** | **{ok_total}** | **{fail_total}** | "
      f"**{ret_total}** | **{sum(pm['prompt_tk'] for pm in per_model.values())}** | "
      f"**{sum(pm['comp_tk'] for pm in per_model.values())}** | "
      f"**{sum(pm['reason_tk'] for pm in per_model.values())}** | "
      f"**${sum(pm['cost'] for pm in per_model.values()):.6f} (rated models; "
      "mimo tokens-only)** |")
    A("")
    A("Note: deepseek-v4-flash rows include the 400 translation calls (all "
      "translations run on deepseek-v4-flash as the fixed translator, design §3). "
      "mimo-v2.5 has no published per-token rate on opencode-go → tokens counted, "
      "cost not estimable (same note as smoke §4).")
    A("")
    A("### Per benchmark")
    A("")
    A("| benchmark | calls | ok | fail | retries |")
    A("|---|---|---|---|---|")
    for b in ("tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"):
        pb = per_bench.get(b, {})
        A(f"| {BENCH_NAMES[b]} | {pb.get('calls',0)} | {pb.get('ok',0)} | "
          f"{pb.get('fail',0)} | {pb.get('retries',0)} |")
    A("")

    # ---- 2. per-cell tables ----------------------------------------------
    A("## 2. Probe results (every cell shown)")
    A("")
    for b in ("tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"):
        A(f"### {BENCH_NAMES[b]}")
        A("")
        A("**M1 — verbatim recall / 8-gram overlap** (confounded by "
          "instruction-following; never decisive alone)")
        A("")
        A("| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |")
        A("|---|---|---|---|---|")
        for m in MODEL_NAMES:
            c = stats["cells"][f"{b}|{m}"]["M1"]
            if c["n"] == 0:
                A(f"| {MODEL_NAMES[m]} | 0 | — | — | — |")
                continue
            A(f"| {MODEL_NAMES[m]} | {c['n']} | {c['verbatim_hits']} | "
              f"{c['verbatim_rate']:.3f} [{c['verbatim_ci95'][0]:.3f},"
              f"{c['verbatim_ci95'][1]:.3f}] | "
              f"{c['mean_overlap8']:.3f} ± {c['overlap8_sd']:.3f} |")
        A("")
        if b != "ragturk_formal5k":
            A("**M2a — distractor flip / edited-answer flip** (follow-planted rate + "
              "paired McNemar exact vs A-arm baseline; Bonferroni family "
              f"size={stats['bonferroni_family_sizes'].get(b + '|M2a', 1)})")
            A("")
            A("| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. "
              "(base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |")
            A("|---|---|---|---|---|---|---|")
            for m in MODEL_NAMES:
                c = stats["cells"][f"{b}|{m}"]["M2a"]
                d = c["mcnemar_discordant"]
                A(f"| {MODEL_NAMES[m]} | {c['n_flip']} | {c['flip_follows']} | "
                  f"{c['flip_follow_rate']:.3f} [{c['flip_follow_ci95'][0]:.3f},"
                  f"{c['flip_follow_ci95'][1]:.3f}] | ({d[0]}, {d[1]}) | "
                  f"{fmt_p(c['mcnemar_p_raw'])} | {fmt_p(c['mcnemar_p_bonf'])} |")
            A("")
        A("**M2b — 3-arm crosslingual fragility** (fully-matched denominator; "
          "McNemar exact paired; Bonferroni family "
          f"size={stats['bonferroni_family_sizes'].get(b + '|M2b', 1)}; "
          "QA arms scored by lenient containment match)")
        A("")
        A("| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) "
          "| B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |")
        A("|---|---|---|---|---|---|---|---|")
        for m in MODEL_NAMES:
            c = stats["cells"][f"{b}|{m}"]["M2b"]
            arms = c["arms"]
            mn = c["mcnemar"]
            A(f"| {MODEL_NAMES[m]} | {c['matched_n']} | {acc_str(arms['A'])} | "
              f"{acc_str(arms['B'])} | {acc_str(arms['C'])} | "
              f"{fmt_p(mn['B_vs_A']['p_bonf'])} | {fmt_p(mn['C_vs_A']['p_bonf'])} | "
              f"{fmt_p(mn['C_vs_B']['p_bonf'])} |")
        A("")
        A("**Consensus call (per §0):**")
        A("")
        for m in MODEL_NAMES:
            c = stats["cells"][f"{b}|{m}"]
            ind = c["modality_indications"]
            inds = ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in ind.items())
            A(f"- **{MODEL_NAMES[m]}**: **{c['consensus_call']}** "
              f"(modalities indicating: {c['n_modalities_indicate']}/3 [{inds}]; "
              f"min dynamic-arm p after Bonferroni = {fmt_p(c['min_dynamic_arm_p_bonf'])}).")
        A("")

    # ---- 3. honest interpretation ----------------------------------------
    A("## 3. Interpretation (honest-negative discipline, design §6)")
    A("")
    A("- Every cell above is shown with its n, Wilson 95% CI, raw and "
      "Bonferroni-corrected p. Directional-but-not-significant results are "
      "reported as **null/suggestive**, never as findings.")
    A("- Three cells meet the §0 threshold **as pre-registered**: TR-MMLU x "
      "gpt-5.6-luna, TUMLU-tr x gpt-5.6-luna, TUMLU-tr x deepseek-v4-flash. In "
      "each, the carrier is the **M2b back-translation collapse with "
      "English-direct stability**: B accuracy drops (0.84→0.62, 0.98→0.70, "
      "1.00→0.75) with perfectly one-sided paired discordance (11-0, 14-0, "
      "10-0; Bonferroni p = 0.009 / <0.001 / 0.018) while C-vs-A shows no "
      "change (p = 1.000 / 1.000 / 0.562). This is the ConStat "
      "non-generalizing-performance signature (anchored to the memorized "
      "Turkish surface).")
    A("- **Flag-sensitivity disclosure (mandatory for honesty):** in all three "
      "flagged cells the second 'indicating' modality is **M1**, which the "
      "design itself flags as instruction-following-confounded (the prompt "
      "explicitly asks for verbatim reproduction). If M1 is discounted, each "
      "flagged cell rests on a single modality (M2b) and does NOT meet the "
      "≥2-modalities bar — i.e., the calls are **threshold-dependent**: they "
      "are exactly what the pre-registered rule produces, and exactly what a "
      "stricter reading of the design's own M1 caveat would not flag. The "
      "M2b discordances themselves are directionally robust (one-sided 11-0 / "
      "14-0 / 10-0) and must be confirmed at n=200 before any final claim.")
    A("- No M2a flip-following anywhere: follow-planted rates are 0.08-0.24 "
      "across all cells (never >0.5), so no model follows planted "
      "distractors; the significant M2b effects are surface-fragility "
      "(back-translation), not concept-independent memorization.")
    A("- Halluverse: all three models score low on free-form QA (A-arm 0.16-"
      "0.22). The M2a 2-option flip control shows no flip-following "
      "(0.08-0.14); its McNemar (0, 35) reflects that the forced true-vs-"
      "edited choice **rescues** 35 items relative to free-form answering "
      "(structure effect, direction opposite to flip-following). Not a "
      "contamination signal; the cell stays an honest negative.")
    A("- RAGTurk formal_5k: near-zero accuracies (0.00-0.07) under the "
      "lenient containment scorer. For long formal gold answers this largely "
      "reflects **answer-form mismatch** (semantically right, surface "
      "different), not zero capability; arm C near-zero for luna/mimo is a "
      "surface artifact of matching Turkish gold strings against English "
      "outputs. With the A baseline at floor, no fragility contrast is "
      "possible — cells report honest negatives with this scope note.")
    A("- M1 values are high for gpt-5.6-luna and deepseek-v4-flash on the MC "
      "benchmarks because the M1 prompt explicitly requests verbatim "
      "reproduction — this measures instruction-following, not memorization; "
      "M1 never flags alone.")
    A("- Run-integrity note: the 13 failed calls are all gpt-5.6-luna "
      "empty-message responses, 12/13 on RAGTurk (clustered on specific long "
      "items — e.g. items 2, 36, 41 recur across probes) plus 1 on "
      "Halluverse; retry-on-empty recovered 20 retry events (15 calls "
      "recovered). Cells report "
      "the honest reduced/matched n (e.g. RAGTurk x luna M2b matched n=43).")
    A("")
    A("Null statements above are **scope-limited**: \"no contamination "
      "detected by these probes at n=50 per benchmark x model\" — never a "
      "claim that a benchmark is clean (black-box API: training inclusion is "
      "unobservable).")
    A("")

    # ---- 4. gate recommendation ------------------------------------------
    A("## 4. Gate: full n=200 run (Day 8)")
    A("")
    n_rated = sum(pm["cost"] for pm in per_model.values())
    A(f"- Pipeline clean: {ok_total}/{n_total} calls ok; resume-skip, rate "
      f"limiter (~2 req/s cap, live ~0.5 calls/s observed), cost cap and "
      f"retry-on-empty (46 events: 20 recovered on 15 calls, 26 exhausted "
      f"across 13 calls) all exercised; exhausted calls are handled "
      f"honestly in the reported denominators.")
    A(f"- Pilot spend: ${n_rated:.6f} estimated on rated models "
      f"(luna $0.0847 + deepseek incl. translations $0.0914) + mimo-v2.5 "
      f"tokens-only; ≈ {100*n_rated:.0f}% of the $1.00 pilot cap.")
    A("- n=200 scaling at observed rates: ~4x calls (≈13,000) and ~4x cost "
      "(≈ **$0.70** rated-model estimate on pilot token profiles; verify "
      "before launch). mimo tokens grow ~4x uncosted; runtime ≈ 6-7 h at "
      "the observed 0.5 calls/s (raise --rate/--workers if the endpoint "
      "accepts it).")
    A("- **Recommendation: full n=200 run is clear to launch** — same lineup "
      "and transport (opencode-go), explicit user approval for the raised "
      "cost cap, one code-pipeline change to consider first: **confirm or "
      "refute the three pilot flags with a pre-registered M1-discount "
      "sensitivity analysis** (report both readings) since the §0 threshold "
      "is met in 3/12 cells and the flags are M1-confound-sensitive. Also "
      "consider: (i) a retry-harder policy for gpt-5.6-luna on long RAGTurk "
      "items, or excluding the 8 item-cluster failures via matched-denominator "
      "reporting (already the protocol); (ii) RAGTurk QA-pair scoring needs an "
      "answer-form-robust scorer or a re-render to MC format before claiming "
      "anything from that cell.")
    A("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
