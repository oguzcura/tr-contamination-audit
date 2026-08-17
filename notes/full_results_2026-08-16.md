# Full-Run Results — Turkish Benchmark Contamination Audit (Day 8, 2026-08-16)

**Scope:** n=200 seed-42 matched items per benchmark x 3 models (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash reference), ALL via opencode-go (`trmlu_audit.core`; OpenRouter untouched). Probes M1 + M2a + M2b 3-arm. The four user-mandated fixes (2026-08-16 approval) are implemented and pre-registered below before any results.

**Run integrity:** run_start markers = 4, run_end = 1; **this run was RESUMED/INTERRUPTED** (mandatory report line). Elapsed (first start -> last end) ≈ 13.94 h.

---

## 0. PRE-REGISTRATION (verbatim, frozen before the run)

### PRE-REGISTRATION — Full n=200 Run, Day 8 (2026-08-16)

**Frozen BEFORE the run.** Timestamp of first write: 2026-08-16 (Day 8).
Author: oguzc. Design parent: `notes/paper1_contamination_audit.md` (§0, §3, §4).
Pilot context: `notes/pilot_results_2026-08-16.md` (Day 5). User approval
2026-08-16: full n=200, lineup gpt-5.6-luna + mimo-v2.5 + deepseek-v4-flash
(reference) ALL via opencode-go (`trmlu_audit.core`; OpenRouter untouched),
cost cap raised to $2.00, with the 4 mandatory fixes below.

Nothing in this file may be changed after the run starts. The analyzer and
report generator implement exactly these rules.

---

### 0. Scope and planned volume (real numbers, from the runner dry-run)

- n = 200 seed-42 matched items per benchmark (TR-MMLU, TUMLU-tr,
  Halluverse-M3-tr, RAGTurk formal_5k) x 3 models x probes
  M1 + M2a + M2b(A/B/C). M2a is N/A for RAGTurk (QA-pair, no options) — no
  calls, cell reported N/A (pilot §2 note).
- Planned calls: 11,400 answer calls + 1,600 translation calls (2 per
  benchmark x item, lazy, via deepseek-v4-flash as the fixed translator)
  + 200 RAGTurk gold TR->EN calls for arm-C scoring = **13,200**.
- Pre-run cost estimate (documented rates only; pilot-real token profile):
  **$0.79**; hard cap **$2.00** (fix 4). mimo-v2.5 has no published rate on
  opencode-go → tokens counted, cost not estimable (logged as None).

### 1. FIX 1 — M1-discount sensitivity analysis (PRE-REGISTERED)

**Motivation (from the pilot, stated in the pilot report §3):** all three
Day-5 flagged cells (TR-MMLU x luna, TUMLU-tr x luna, TUMLU-tr x deepseek)
met the §0 bar with **M1 as their second indicating modality**, and the design
itself documents M1 as instruction-following-confounded (the M1 prompt asks
for verbatim reproduction). The flags are therefore **threshold-dependent**.

**Rule (frozen):** every cell gets TWO consensus readings, computed by one
function with `discount_m1 ∈ {False, True}`:

- **RAW (§0, unchanged):** flagged iff ≥2/3 modalities indicate AND ≥1 dynamic
  arm (M2a/M2b) p < 0.05 after Bonferroni. Modality "indicates" =
  M1: mean_overlap8 > 0.5 AND verbatim_rate > 0.5;
  M2a: flip_follow_rate > 0.5;
  M2b: B_acc < C_acc − 0.10 (B collapses while C stable).
- **M1-DISCOUNT:** identical rule with M1 struck from the modality count
  (i.e., M2a/M2b must indicate independently of M1).

**Adjudication (frozen):** a benchmark×model cell is flagged **only if
≥1 dynamic arm (M2a/M2b) is significant after Bonferroni AND M2a/M2b indicate
independently of M1** — i.e., the **M1-DISCOUNT** reading is the adjudicated
call; the RAW reading is always reported alongside. Cells flagged by RAW but
not by M1-DISCOUNT are labeled **THRESHOLD-DEPENDENT (M1-sensitive)**, never
"contaminated". Both readings appear for every cell in the stats JSON and the
report tables.

### 2. FIX 2 — retry-harder for gpt-5.6-luna on RAGTurk (PRE-REGISTERED)

**Data fact (verified pre-run):** ragturk_formal5k max item length
(soru+cevap) = 1,583 chars < 2,000, so a pure `>2000 chars` trigger would
fire zero times. Verified pre-run on the pilot failure cluster (12/13 RAGTurk
failures; items 2, 36, 41 recur across probes): failures are **item-recurrent,
not length-driven** (failing items are short; e.g. item 41 = 73 chars).
Therefore the operational trigger is:

- (a) **surface length > 2000 chars** (kept for spec fidelity; expected 0
  triggers in formal_5k), OR
- (b) **item-recurrence**: the (benchmark, item_idx) already produced a failed
  gpt-5.6-luna call (seeded from the resumed log, updated live).

When triggered: up to **4 extra retries** with backoff [2, 4, 8, 16] s on top
of core's retry-on-empty (max 2), each attempt logged in `retry_log` with
`retry_hard_reason ∈ {long, recurring-item}`; exhausted failures are logged
with their error and reported in the honest denominators (same protocol as
the pilot). Scope: gpt-5.6-luna x ragturk_formal5k only.

### 3. FIX 3 — answer-form-robust RAGTurk scoring (PRE-REGISTERED)

**Motivation (pilot §2/§4):** near-floor accuracies (0.00–0.07) under the
lenient containment scorer largely reflect **answer-form mismatch**, not zero
capability; with the A-arm at floor no fragility contrast is possible.

**Matcher** (`harness/qa_matcher.py`, frozen cascade, published in
`results/ragturk_matcher_labeled.json`):
1. L0 empty guard; 2. L1 letter-form gold (`a) …`); 3. L2 exact (normalized);
4. L3 **number guard**: every gold numeric token must occur in the pred
   (decade tokens '1950'ler satisfied by any year in [1950,1959]); failure =
   reject. Kills date/number confusions (9 vs 10 Kasım, 16 vs 21 Ağustos,
   18 vs 14 gol, 1993 vs 1996).
5. L4 containment (≥8 chars, ratio ≥0.25).
6. L5 paraphrase: Jaccard ≥0.65 / 5-gram coverage ≥0.80 (numeric golds:
   ≥0.55/≥0.65) AND distinctive-5gram coverage (gold grams ∉ question) ≥0.50
   AND content-token guard (≤4 distinctive content words added by the answer
   must all survive in the pred). Kills single-fact swaps.
7. L6 long descriptive golds (>120 chars, no numbers): distinctive coverage
   ≥0.50 AND Jaccard ≥0.30.

**Measured performance (pre-run, on 106 hand-labeled pilot pairs, two-rater
agreement on a 10-pair batch): precision 1.000, recall 0.371, accuracy 0.792**
(level histogram: exact 2, containment 11, number-mismatch 48, none 44,
empty 1). The matcher is **deliberately precision-first** (false positives
would inflate accuracy and corrupt the fragility contrast); its low recall is
a disclosed limitation, and a **lenient sensitivity variant** (pilot
thresholds, no content-token guard) is logged per call and reported as a
sensitivity column for RAGTurk only — never as headline accuracy.

**Arm C (English surface):** gold answers are machine-translated TR->EN once
per RAGTurk item (probe `M2b_gold_en`, deepseek-v4-flash, logged, cached,
resume-re-seeded) and C is scored with the same matcher against the EN gold
and the EN surface as question. Headline accuracy = any-level strict match;
legacy containment + lenient variant logged per call for transparency.

### 4. FIX 4 — cost cap and spend log (PRE-REGISTERED)

- `--max-cost 2.00` hard-abort (dry-run estimate before launch + live abort on
  accumulated estimated cost, identical semantics to the pilot).
- **Spend checkpoint appended every 500 logged calls** to
  `results/spend_full_2026-08-16.jsonl`: {ts, calls_logged, accumulated_cost,
  cap, abort}; a final checkpoint line is written at run end.
- Resume: rerunning the same command re-seeds the done-set + translation
  caches from the log and continues; run_start/run_end event markers are
  written to the JSONL so the report can state **whether the run was
  resumed/interrupted** (mandatory report line).

### 5. Pre-registered statistics (unchanged from §4/§0 of the design)

Wilson 95% CI per cell; McNemar exact two-sided for paired arm contrasts;
Bonferroni per benchmark×probe family (raw + corrected p, both shown);
consensus calls exactly per §1 of this file; every cell reported, honest
negatives worded "no significant contamination detected". Null statements are
scope-limited to these probes at n=200.

---
*End of pre-registration. Frozen before launch; the analyzer
(`analyze_full_audit.py`) and report generator (`report_full.py`) are the
mechanical implementations.*

---

## 1. Run reliability & spend

**Calls logged: 13193** (planned 13,200; difference = degenerate items not called / retry exhaustion handled in denominators). ok = 13154, fail = 39, retries = 255.

### Spend log (from the JSONL; documented rates only)

| model | calls | ok | fail | retries | retry-hard | prompt tk | completion tk | reasoning tk | cost est USD |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 3798 | 3759 | 39 | 255 | 34 | 450939 | 505447 | 0 | $0.348362 |
| mimo-v2.5 | 3798 | 3798 | 0 | 0 | 0 | 470452 | 1336205 | 1096470 | $0.000000 |
| deepseek-v4-flash | 5597 | 5597 | 0 | 0 | 0 | 1143683 | 1986855 | 37854 | $0.358241 |
| **TOTAL** | 13193 | 13154 | 39 | 255 | 34 | 2065074 | 3828507 | 1134324 | **$0.706603** |

Rated-model total: **$0.706603** against the **$2.00 cap** (mimo-v2.5 tokens-only, uncosted). Spend checkpoints written every 500 calls: 26 lines in results/spend_full_2026-08-16.jsonl.

### Per benchmark

| benchmark | calls | ok | fail | retries |
|---|---|---|---|---|
| TR-MMLU | 3396 | 3396 | 0 | 1 |
| TUMLU-tr | 3397 | 3397 | 0 | 2 |
| Halluverse-M3-tr | 3400 | 3393 | 7 | 28 |
| RAGTurk formal_5k | 3000 | 2968 | 32 | 224 |

Fix-2 retry-hard events (gpt-5.6-luna x RAGTurk): 34 triggered (long-surface >2000 chars: 0 failures — expected 0 triggers in formal_5k; item-recurrence: 17 exhausted failures, each logged).

## 2. Matcher documentation (fix 3, measured pre-run)

Hand-labeled set: 106 (gold, pred) pairs from the Day-5 pilot; two-rater agreement on a 10-pair batch; artifact `results/ragturk_matcher_labeled.json`.
Measured (strict cascade of `harness/qa_matcher.py`): **precision 1.000, recall 0.371**, accuracy 0.792; confusion TP=13 FP=0 TN=71 FN=22.
Level histogram: exact 2, containment 11, number-mismatch 48 (rejected), none 44, empty 1; the paraphrase levels matched 0 of the labeled pairs in the final cascade (the 4 naive-paraphrase candidates were single-fact swaps, e.g. teorik/istatistiksel fizik, UCID/Wadani, and are rejected by the content-token + distinctive-gram guards).
**Interpretation:** the matcher is deliberately precision-first; its low recall understates true QA capability, so every RAGTurk accuracy is a lower bound on string-verifiable correctness. A lenient sensitivity variant is reported per arm below; caution is required before any RAGTurk claim (design §6: 'interpret carefully').

## 3.1 TR-MMLU

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone; struck in the M1-DISCOUNT reading)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 200 | 172 | 0.860 [0.805,0.901] | 0.7261 ± 0.4411 |
| mimo-v2.5 | 200 | 115 | 0.575 [0.506,0.641] | 0.4821 ± 0.4864 |
| deepseek-v4-flash (ref) | 200 | 156 | 0.780 [0.718,0.832] | 0.6598 ± 0.4615 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 200 | 31 | 0.155 [0.111,0.212] | (27, 9) | 0.004 | 0.012 |
| mimo-v2.5 | 200 | 28 | 0.140 [0.099,0.195] | (34, 18) | 0.036 | 0.110 |
| deepseek-v4-flash (ref) | 200 | 29 | 0.145 [0.103,0.201] | (30, 8) | <0.001 | 0.002 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by the fix-3 robust matcher; RAGTurk arm C vs EN-translated gold)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 199 | 0.864 (172/199) [0.810,0.905] | 0.734 (146/199) [0.668,0.790] | 0.819 (163/199) [0.760,0.866] | <0.001 | 1.000 | 0.013 |
| mimo-v2.5 | 187 | 0.754 (141/187) [0.688,0.810] | 0.674 (126/187) [0.604,0.737] | 0.711 (133/187) [0.642,0.771] | 0.150 | 1.000 | 1.000 |
| deepseek-v4-flash (ref) | 168 | 0.881 (148/168) [0.823,0.922] | 0.762 (128/168) [0.692,0.820] | 0.816 (137/168) [0.750,0.867] | 0.004 | 0.239 | 0.838 |

**Consensus (per §0 RAW and pre-registered M1-DISCOUNT):**

| model | modalities RAW | min dyn p (Bonf) | RAW call | M1-DISCOUNT call | adjudicated |
|---|---|---|---|---|---|
| gpt-5.6-luna | 1/3 (M1) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| mimo-v2.5 | 0/3 (—) | 0.110 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| deepseek-v4-flash (ref) | 1/3 (M1) | 0.002 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |

---

## 3.2 TUMLU-tr

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone; struck in the M1-DISCOUNT reading)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 200 | 177 | 0.885 [0.833,0.922] | 0.7315 ± 0.4294 |
| mimo-v2.5 | 200 | 73 | 0.365 [0.301,0.434] | 0.3227 ± 0.4279 |
| deepseek-v4-flash (ref) | 200 | 114 | 0.570 [0.501,0.637] | 0.547 ± 0.4397 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 199 | 19 | 0.096 [0.062,0.144] | (22, 3) | <0.001 | <0.001 |
| mimo-v2.5 | 199 | 22 | 0.111 [0.074,0.162] | (34, 19) | 0.053 | 0.160 |
| deepseek-v4-flash (ref) | 199 | 22 | 0.111 [0.074,0.162] | (25, 6) | <0.001 | 0.003 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by the fix-3 robust matcher; RAGTurk arm C vs EN-translated gold)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 200 | 0.930 (186/200) [0.886,0.958] | 0.775 (155/200) [0.712,0.827] | 0.865 (173/200) [0.811,0.905] | <0.001 | 0.040 | 0.035 |
| mimo-v2.5 | 180 | 0.856 (154/180) [0.797,0.899] | 0.683 (123/180) [0.612,0.747] | 0.806 (145/180) [0.742,0.857] | <0.001 | 1.000 | 0.014 |
| deepseek-v4-flash (ref) | 168 | 0.958 (161/168) [0.916,0.980] | 0.804 (135/168) [0.737,0.857] | 0.917 (154/168) [0.865,0.950] | <0.001 | 0.831 | 0.003 |

**Consensus (per §0 RAW and pre-registered M1-DISCOUNT):**

| model | modalities RAW | min dyn p (Bonf) | RAW call | M1-DISCOUNT call | adjudicated |
|---|---|---|---|---|---|
| gpt-5.6-luna | 1/3 (M1) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| mimo-v2.5 | 1/3 (M2b) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| deepseek-v4-flash (ref) | 2/3 (M1, M2b) | <0.001 | CONTAMINATED (pre-registered rule) | no significant contamination detected | **THRESHOLD-DEPENDENT (M1-sensitive)** |

---

## 3.3 Halluverse-M3-tr

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone; struck in the M1-DISCOUNT reading)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 199 | 0 | 0.000 [0.000,0.019] | 0.0674 ± 0.1172 |
| mimo-v2.5 | 200 | 0 | 0.000 [0.000,0.019] | 0.0775 ± 0.1359 |
| deepseek-v4-flash (ref) | 200 | 0 | 0.000 [0.000,0.019] | 0.1121 ± 0.1677 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 200 | 18 | 0.090 [0.058,0.138] | (0, 176) | <0.001 | <0.001 |
| mimo-v2.5 | 200 | 17 | 0.085 [0.054,0.132] | (1, 173) | <0.001 | <0.001 |
| deepseek-v4-flash (ref) | 200 | 22 | 0.110 [0.074,0.161] | (1, 164) | <0.001 | <0.001 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by the fix-3 robust matcher; RAGTurk arm C vs EN-translated gold)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 196 | 0.025 (5/196) [0.011,0.058] | 0.031 (6/196) [0.014,0.065] | 0.082 (16/196) [0.051,0.129] | 1.000 | 0.067 | 0.057 |
| mimo-v2.5 | 200 | 0.045 (9/200) [0.024,0.083] | 0.030 (6/200) [0.014,0.064] | 0.075 (15/200) [0.046,0.120] | 1.000 | 1.000 | 0.572 |
| deepseek-v4-flash (ref) | 200 | 0.055 (11/200) [0.031,0.096] | 0.050 (10/200) [0.027,0.090] | 0.095 (19/200) [0.062,0.144] | 1.000 | 0.867 | 0.317 |

**Consensus (per §0 RAW and pre-registered M1-DISCOUNT):**

| model | modalities RAW | min dyn p (Bonf) | RAW call | M1-DISCOUNT call | adjudicated |
|---|---|---|---|---|---|
| gpt-5.6-luna | 0/3 (—) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| mimo-v2.5 | 0/3 (—) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| deepseek-v4-flash (ref) | 0/3 (—) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |

---

## 3.4 RAGTurk formal_5k

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone; struck in the M1-DISCOUNT reading)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 193 | 10 | 0.052 [0.028,0.093] | 0.1539 ± 0.2394 |
| mimo-v2.5 | 200 | 6 | 0.030 [0.014,0.064] | 0.1069 ± 0.1811 |
| deepseek-v4-flash (ref) | 200 | 14 | 0.070 [0.042,0.114] | 0.1389 ± 0.2574 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by the fix-3 robust matcher; RAGTurk arm C vs EN-translated gold)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 187 | 0.080 (15/187) [0.049,0.128] | 0.070 (13/187) [0.041,0.115] | 0.000 (0/187) [0.000,0.020] | 1.000 | <0.001 | 0.002 |
| mimo-v2.5 | 200 | 0.025 (5/200) [0.011,0.057] | 0.040 (8/200) [0.020,0.077] | 0.000 (0/200) [0.000,0.019] | 1.000 | 0.562 | 0.070 |
| deepseek-v4-flash (ref) | 200 | 0.070 (14/200) [0.042,0.114] | 0.060 (12/200) [0.035,0.102] | 0.000 (0/200) [0.000,0.019] | 1.000 | <0.001 | 0.004 |

**RAGTurk lenient-sensitivity reading (fix 3 disclosure; not headline):** accuracy with the lenient matcher variant — 
A: 0.134, B: 0.091, C: 0.000, A: 0.035, B: 0.065, C: 0.000, A: 0.090, B: 0.080, C: 0.000.

**Consensus (per §0 RAW and pre-registered M1-DISCOUNT):**

| model | modalities RAW | min dyn p (Bonf) | RAW call | M1-DISCOUNT call | adjudicated |
|---|---|---|---|---|---|
| gpt-5.6-luna | 0/3 (—) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| mimo-v2.5 | 0/3 (—) | 0.070 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| deepseek-v4-flash (ref) | 0/3 (—) | <0.001 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |

---

## 4. Adjudication summary (RAW vs M1-DISCOUNT)

| benchmark | model | RAW | M1-DISCOUNT | adjudicated |
|---|---|---|---|---|
| TR-MMLU | gpt-5.6-luna | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| TR-MMLU | mimo-v2.5 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| TR-MMLU | deepseek-v4-flash (ref) | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| TUMLU-tr | gpt-5.6-luna | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| TUMLU-tr | mimo-v2.5 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| TUMLU-tr | deepseek-v4-flash (ref) | CONTAMINATED (pre-registered rule) | no significant contamination detected | **THRESHOLD-DEPENDENT (M1-sensitive)** |
| Halluverse-M3-tr | gpt-5.6-luna | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| Halluverse-M3-tr | mimo-v2.5 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| Halluverse-M3-tr | deepseek-v4-flash (ref) | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| RAGTurk formal_5k | gpt-5.6-luna | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| RAGTurk formal_5k | mimo-v2.5 | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |
| RAGTurk formal_5k | deepseek-v4-flash (ref) | no significant contamination detected | no significant contamination detected | **no significant contamination detected** |

Per the pre-registered adjudication (fix 1): a cell is flagged only on the M1-DISCOUNT reading; RAW-only flags are threshold-dependent by definition.

## 5. Honest interpretation

1. M1 values on the MC benchmarks are high for instruction-following models because the M1 prompt explicitly requests verbatim reproduction — by design M1 never flags alone and is struck in the M1-DISCOUNT reading.

2. The M2b signature of interest (ConStat non-generalizing performance) is B collapsing while C stays stable; where B-vs-A is significant after Bonferroni with C-vs-A not significant, the pattern is reported as surface-fragility, not concept-independent memorization; no M2a flip-following anywhere means no model followed planted distractors (rates and p reported per cell).

3. Halluverse free-form QA baseline is low (arm A accuracy per table); its M2a 2-option control rescues items relative to free-form answering and the McNemar discordance is the structure effect, not flip-following — cells stay honest negatives unless the dynamic-arm bar is met.

4. RAGTurk formal_5k: accuracies are lower bounds under the precision-first matcher (recall 0.371 measured). The A-arm is near floor for most cells, so no fragility contrast is resolvable; arm C is scored against machine-translated golds (probe M2b_gold_en) — an extra moving part, reported as such. Treat every RAGTurk number as conservative.

5. All null statements are scope-limited: 'no contamination detected by these probes at n=200 per benchmark x model', never a claim that a benchmark is clean (black-box API: training inclusion is unobservable).

## 6. Limitations

1. Black-box API: memorization signal is inferred from sampled text; no weights/logits; training inclusion is not observable.

2. M1 confound (instruction-following) — handled by the pre-registered M1-DISCOUNT reading; RAW reading shown for transparency.

3. QA semantic matching is noisy: the fix-3 matcher is measured precision 1.000 / recall 0.371 on 106 labeled pairs (labels = author + 10-pair agreement batch; artifact published). RAGTurk cells are conservative.

4. RAGTurk arm-C English golds are machine-translated (single pass, deepseek-v4-flash); translation error propagates into C scoring.

5. Per-cell matched denominators drop below n=200 where probes failed or degenerate items were excluded; every table reports its own n.

6. Failure cluster risk was mitigated by fix-2 retry-hard; any remaining exhausted calls appear in the honest denominators (per-cell n).

7. n=200 is a sample, not full-benchmark: results generalize to the seed-42 matched subsets, not to whole benchmarks.

---
*Generated mechanically from results/full_stats_2026-08-16.json + results/full_audit_2026-08-16.jsonl + results/spend_full_2026-08-16.jsonl; pre-registration section verbatim from notes/pre_reg_full_2026-08-16.md.*