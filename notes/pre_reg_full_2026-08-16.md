# PRE-REGISTRATION — Full n=200 Run, Day 8 (2026-08-16)

**Frozen BEFORE the run.** Timestamp of first write: 2026-08-16 (Day 8).
Author: oguzc. Design parent: `notes/paper1_contamination_audit.md` (§0, §3, §4).
Pilot context: `notes/pilot_results_2026-08-16.md` (Day 5). User approval
2026-08-16: full n=200, lineup gpt-5.6-luna + mimo-v2.5 + deepseek-v4-flash
(reference) ALL via opencode-go (`trmlu_audit.core`; OpenRouter untouched),
cost cap raised to $2.00, with the 4 mandatory fixes below.

Nothing in this file may be changed after the run starts. The analyzer and
report generator implement exactly these rules.

---

## 0. Scope and planned volume (real numbers, from the runner dry-run)

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

## 1. FIX 1 — M1-discount sensitivity analysis (PRE-REGISTERED)

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

## 2. FIX 2 — retry-harder for gpt-5.6-luna on RAGTurk (PRE-REGISTERED)

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

## 3. FIX 3 — answer-form-robust RAGTurk scoring (PRE-REGISTERED)

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

## 4. FIX 4 — cost cap and spend log (PRE-REGISTERED)

- `--max-cost 2.00` hard-abort (dry-run estimate before launch + live abort on
  accumulated estimated cost, identical semantics to the pilot).
- **Spend checkpoint appended every 500 logged calls** to
  `results/spend_full_2026-08-16.jsonl`: {ts, calls_logged, accumulated_cost,
  cap, abort}; a final checkpoint line is written at run end.
- Resume: rerunning the same command re-seeds the done-set + translation
  caches from the log and continues; run_start/run_end event markers are
  written to the JSONL so the report can state **whether the run was
  resumed/interrupted** (mandatory report line).

## 5. Pre-registered statistics (unchanged from §4/§0 of the design)

Wilson 95% CI per cell; McNemar exact two-sided for paired arm contrasts;
Bonferroni per benchmark×probe family (raw + corrected p, both shown);
consensus calls exactly per §1 of this file; every cell reported, honest
negatives worded "no significant contamination detected". Null statements are
scope-limited to these probes at n=200.

---
*End of pre-registration. Frozen before launch; the analyzer
(`analyze_full_audit.py`) and report generator (`report_full.py`) are the
mechanical implementations.*