# Paper 2 — NeurIPS 2026 Workshop Submission Skeleton
**Subject:** Multi-probe, API-only contamination audit of TR-MMLU and the 2025–2026 Turkish evaluation wave (paper 2; supersedes the single-benchmark paper-1 scale3arm run)
**Prepared:** 2026-08-16 · **Updated:** 2026-08-17 (full n=200 numbers) · **Target deadline:** Aug 29, 2026 AoE (JUDGe / TAI-Eval / AIDaR — all live-verified)
**Author:** oguzc (independent, high-school, Türkiye) · **Hardware:** none — pure API black-box via opencode-go

> **SOURCING RULE (do not break):** every number in this skeleton is now from the
> full n=200 run (seed 42, completed 2026-08-16/17;
> `results/full_stats_2026-08-16.json`), which replaced every earlier
> `[PILOT, n=50]` tag on 2026-08-17. The pre-registered adjudication
> (M1-DISCOUNT) is an honest negative for all 12 cells: 0/12 flags; one RAW-only
> cell (TUMLU-tr × deepseek-v4-flash) is reported as THRESHOLD-DEPENDENT
> (M1-sensitive), never "contaminated". A positive "contamination found" claim
> remains forbidden unless the frozen decision rule is met post-discounting.

---

## 1. TITLE CANDIDATES (3)

1. **RECOMMENDED — "Contamination Signals, Not Verdicts: A Pre-Registered Multi-Probe Audit of Turkish LLM Benchmarks with a Threshold-Sensitivity Protocol"**
   - Why: leads with the paper's honest posture (signals ≠ verdicts), names the pre-registration and the M1-discount sensitivity analysis — the paper's two method contributions; maps directly onto JUDGe's "evaluator validity" theme (how do we know a benchmark measures what we intend?).
2. "How Contaminated Are Turkish LLM Benchmarks? A Pre-Registered, API-Only Audit of TR-MMLU and the 2025–2026 Turkish Evaluation Wave"
   - Brand continuity with paper 1's working title; question-form title; safe for the audit/benchmark-trust track.
3. "Memorization or Capability? Disambiguating Contamination Signals in Four Turkish Benchmarks with Pre-Registered Multi-Probe Auditing"
   - Method-flavored; emphasizes the surface-fragility vs. concept-generalization disambiguation (ConStat's non-generalizing-performance operationalization).

---

## 2. DRAFT ABSTRACT (150–180 words; every number tagged)

> **NUMBERS STATUS (remove before submission):** all numbers below are from the
> full n=200 run (seed 42, completed 2026-08-16/17; source
> `results/full_stats_2026-08-16.json`), replacing every `[PILOT, n=50]` tag.
> The full run is null after pre-registered adjudication (0/12 cells flag under
> M1-DISCOUNT), so this abstract uses the honest-negative wording drafted in
> §3.4 of this skeleton. Numbers were replaced, not deleted or re-tagged.

Public benchmarks risk contamination: models may memorize benchmark surfaces rather than generalize [ConStat 2405.16281]. Turkish — a low-resource language whose benchmarks (TR-MMLU, public since 2024-12-31; TUMLU; Halluverse-M³-tr; RAGTurk) are young and largely machine-translated — is a critical, under-audited case. We present a pre-registered, API-only contamination audit of four Turkish benchmarks across three model families (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash reference), using three independent probe modalities (M1 verbatim-recall, M2a distractor-flip, M2b three-arm cross-lingual fragility), with per-cell Wilson 95% CIs, exact McNemar tests, Bonferroni correction, and a fixed consensus rule (≥2/3 modalities and ≥1 dynamic arm significant). At n=200 (13,193 logged API calls; ≈$0.71 estimated, rated models only), 1/12 cells crossed the raw pre-registered threshold (TUMLU-tr × deepseek-v4-flash), carrying a back-translation-collapse-with-English-stable signature; under the pre-registered M1-discount reading, 0/12 cells do, and that RAW-only flag is reported as threshold-dependent (M1-sensitive). We therefore report **no significant contamination detected by these probes at n=200 per benchmark × model** — a threshold-dependent, honest-negative-compatible result, with every cell shown under both readings.

*Honesty discipline: this abstract never claims "contamination found". It reports the pre-registered honest negative at n=200 (0/12 cells after M1-discount adjudication) and discloses the one RAW-only threshold-dependent cell (TUMLU-tr × deepseek-v4-flash); verdict wording is "no significant contamination detected by these probes at n=200 per benchmark×model".*

---

## 3. SECTION STRUCTURE (with sourcing bullets)

### 1. Introduction
- Benchmark trust is the dominant evaluation-research theme; leaderboards and undisclosed changes erode trust ("Who Verifies the Benchmark?" 2608.07762; "Evaluation Scores Are Perishable" 2607.26191 — validity windows). Source: `paper1_contamination_audit.md` §9 pack A.
- Turkish is the low-resource case where it bites hardest: TR-MMLU public since 2024-12-31 (arXiv 2501.00593) ≈ 19 months of training-data exposure before this audit; an independent quality audit finds 70% of Turkish benchmark datasets fail heuristic quality standards (2504.09714).
- First-of-kind claim, exactly scoped: *first multi-probe, API-only, pre-registered contamination audit of the 2025–2026 Turkish benchmark wave* (not "first contaminated benchmark found"). Evidence file: paper1 §1.2.
- Single-sentence contribution: pre-registered decision rule (paper1 §0) + threshold-sensitivity protocol (M1-discount, pilot §3) + full cell-level disclosure (pilot §2).

### 2. Related Work
- Contamination & benchmark trust: ConStat (non-generalizing performance definition — method spine, §9 ref [C]); GAOKAO-Eval (leakage lets models "game" static benchmarks, 2412.10056); Open Source Data Contamination Report (1–45% overlap across benchmarks, 2310.17589).
- Memorization-probe methodology & limits: OWL (cross-lingual recall of memorized text — M1 precedent, 2505.22945); Memorization Diagnostics Should Be Scale-Aware (single perturbation probes degrade at scale — the multi-probe mandate, 2608.12771); Rephrased-Sample Contamination (paraphrase/translation bypasses string-match decontamination, 2311.04850).
- Low-resource audit precedent: MathShikkha/Bangla (contamination-audited BanglaMATH; 2608.08503) — the genre proof-of-acceptance our paper mirrors for Turkish.
- Turkish resources & quality: TR-MMLU (2501.00593), TUMLU (2502.11020, native-vs-machine-translated tension), quality audit (2504.09714), RAGTurk (2602.03652), Halluverse-M³ (2602.06920), TAS (2603.09996 — paper-only, qualitative anchor, NOT scored; `[CITATION NEEDED]` discipline if quoted).
- All 11 IDs above live-verified via arXiv API (titles match, 2026-08-16). See §4 table.

### 3. Method
- Pre-registration: H0/H1a/H1b + fixed decision rule (flag ⇔ ≥2/3 modalities indicate AND ≥1 dynamic arm p<0.05 after Bonferroni) — frozen 2026-08-16; source: paper1 §0; rule applied exactly in pilot §"Pre-registered decision rule".
- Three probes: M1 verbatim-recall/8-gram overlap (confound-caveated, never decisive alone); M2a distractor/edited-answer flip; M2b 3-arm cross-lingual (TR orig / TR→EN→TR / EN-direct) — the ConStat non-generalizing-performance operationalization; source: paper1 §3 + pilot §Scope.
- Statistics: fully-matched denominators; Wilson 95% CI per cell; McNemar exact two-sided (binomtest, do not double) per paired contrast; Bonferroni per benchmark×probe family; raw + corrected p reported; source: paper1 §4 + pilot §2 tables + `harness/analyze_pilot_audit.py`.
- Transport/scope: pure black-box via opencode-go only (OpenRouter blocked on this account — paper1 §5); models gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash (reference); retry-on-empty logged, denominators report honest reduced n (full run §1: 13,154/13,193 ok; 39 fails — gpt-5.6-luna only, 32 RAGTurk + 7 Halluverse; 255 retries incl. 34 fix-2 retry-hard triggers, 17 RAGTurk items exhausted).

### 4. Results
- **Full-run scorecard (n=200, seed 42):** 12 cells (4 benchmarks × 3 models), every cell shown with n, Wilson CI, raw+Bonferroni p, and both consensus readings — report §3.1–3.4. Raw-threshold flags 1/12: TUMLU-tr×deepseek-v4-flash only (RAW 2/3 modalities: M1, M2b; min dynamic-arm Bonf p < 0.001). Signature: B-arm back-translation collapse with C-arm (EN-direct) stability (TUMLU-tr×deepseek M2b: B vs A Bonf p < 0.001; C vs A p = 0.831; C vs B p = 0.003).
- **Threshold-sensitivity (mandatory secondary reading):** discounting M1 (instruction-following confound, declared in the design itself) leaves the RAW-flagged cell on a single modality → 0/12 flags at n=200; the TUMLU-tr×deepseek cell is THRESHOLD-DEPENDENT (M1-sensitive), not "contaminated". Both readings reported; this is the paper's transparency contribution, not an excuse.
- **Honest negatives featured:** Halluverse (M2a discordance (0,176)/(1,173)/(1,164) per model is a structure-rescue effect — base-wrong→test-ok direction, opposite of flip-following — not a signal); RAGTurk (near-floor strict accuracies, A-arm 0.025–0.080 and C-arm 0.000 across cells, under the precision-first matcher = answer-form mismatch caveat; M2a N/A); M2a flip-follow rates 0.085–0.155 across all 9 MC cells (no model follows planted distractors).
- **Full-run outcome (realized 2026-08-16, option B):** "no significant contamination detected by these probes at n=200 per benchmark×model" under the pre-registered M1-DISCOUNT adjudication — a publishable first-of-kind honest-negative result (paper1 §1.3 honest-negative commitment). One RAW-only flag, TUMLU-tr × deepseek-v4-flash, is disclosed as THRESHOLD-DEPENDENT (M1-sensitive), never "contaminated". All numbers from `results/full_stats_2026-08-16.json`; narrative report: `notes/full_results_2026-08-16.md`.

### 5. Discussion & Limitations
- What a null means, scope-limited: "no contamination detected by these probes at n=…" is never "clean benchmark" (training inclusion unobservable in a black box) — pilot §3 wording to reuse; adopted verbatim for the n=200 verdict.
- Threshold-dependence is a feature, disclosed: the call flips between 1/12 (RAW) and 0/12 (M1-DISCOUNT) solely on the declared M1 caveat; we recommend future audits report both readings (this is the methodological contribution).
- Limitations: per-cell reduced/matched n (e.g., RAGTurk×luna M2b matched n=187; TUMLU-tr×deepseek M2b matched n=168); reasoning-model empty responses (39 fails, RAGTurk- and Halluverse-clustered, gpt-5.6-luna only) handled via matched denominators; RAGTurk scored by the fix-3 precision-first matcher (precision 1.000 / recall 0.371 — every RAGTurk accuracy is a lower bound); M3 corpus-overlap (13-gram, CulturaX-tr/Wikipedia-tr) either included on time (timeline item 4) or explicitly deferred.
- Reproducibility & ethics: seed 42, JSONL logs, cost log ($0.706603 full-run estimate on rated models — luna $0.348362 + deepseek $0.358241, mimo tokens-only — no published rate; $2.00 cap; 26 spend checkpoints), pre-registration timestamp (frozen before the run), full data release plan; verification pass model from `verification_pass_2026-08-16.md` (stats rerun exact-match discipline, 10/10 citation gate) applied to this paper before submission.

---

## 4. RELATED-WORK ANCHORS (all arXiv IDs live-verified 2026-08-16 via export.arxiv.org/api/query)

| # | Ref (first author, year) | arXiv ID | Why cited |
|---|---|---|---|
| 1 | ConStat (Dekoninck, 2024) | 2405.16281 | Contamination = "artificially inflated and non-generalizing benchmark performance"; the operative definition + method spine of M2 [C] |
| 2 | OWL: Cross-Lingual Recall of Memorized Texts (Srivastava, 2025) | 2505.22945 | M1 precedent — models recall memorized surface text across languages (titles 69%, masked entities 6% on never-trained translations) [11] |
| 3 | Memorization Diagnostics Should Be Scale-Aware (Rajput, 2026) | 2608.12771 | Multi-probe mandate: single perturbation probes degrade at scale [10] |
| 4 | GAOKAO-Eval (Lei, 2024) | 2412.10056 | Leakage lets models "game" static benchmarks despite strong scores |
| 5 | Open Source Data Contamination Report (Li, 2023) | 2310.17589 | M3 overlap precedent: 1–45% contamination across benchmarks, rising over time |
| 6 | Rethinking Benchmark and Contamination with Rephrased Samples (Yang, 2023) | 2311.04850 | Paraphrase/translation bypasses string-match decontamination → perturbation probes required |
| 7 | MathShikkha (Ali, 2026) | 2608.08503 | Accepted low-resource (Bangla) contamination-audit precedent — genre proof for Turkish |
| 8 | Who Verifies the Benchmark? (Pardasani, 2026) | 2608.07762 | Benchmark-trust framing (undisclosed changes, selective reporting) |
| 9 | TR-MMLU (Bayram, 2025) | 2501.00593 | Primary dataset: 6200 items/62 sections; public since 2024-12-31 |
| 10 | TUMLU (Isbarov, 2025) | 2502.11020 | Turkic-language benchmark; machine-translation-error tension in construction |
| + | Evaluating Benchmark Quality for Low-Resource Languages: Turkish (Cengiz, 2025) | 2504.09714 | "70% of benchmark datasets fail heuristic quality standards" — construct-validity hook for JUDGe/TAI-Eval |

*Dataset-row context (also verified §9 of paper 1): RAGTurk 2602.03652 (formal_5k, Wikipedia/CulturaX overlap numerator), Halluverse-M³ 2602.06920 (QA_tr, hallucination-edit = natural flipped-option control), TAS 2603.09996 (paper-only, qualitative anchor only).*

---

## 5. TIMELINE — Aug 16 → Aug 29, 2026 AoE (all deadlines live-verified)

- **Aug 17–18 (Mon–Tue):** ✅ full n=200 run completed 2026-08-16/17 — resumed/interrupted (4 run_starts / 1 run_end), ≈13.94 h wall; 13,193 calls logged / 13,154 ok / 39 fail / 255 retries; rated cost $0.7066 vs $2.00 cap; `results/full_stats_2026-08-16.json` + `notes/full_results_2026-08-16.md` written; skeleton updated with full numbers (2026-08-17). Remaining this window: M3 corpus-overlap setup + RAGTurk scorer decision close-out (below).
- **Aug 19 (Wed):** M3 corpus-overlap (13-gram hashes vs CulturaX-tr / Wikipedia-tr) — can run concurrently; RAGTurk scorer decision RESOLVED by pre-registered fix 3 (`harness/qa_matcher.py`, precision-first, lenient sensitivity variant logged per call) — remaining task is to lock the final scope note wording; delete/hide dead `harness/scale3arm.py` trap (verification-pass housekeeping flag).
- **Aug 20–21 (Thu–Fri):** Fill Results: 12-cell scorecard (n, Wilson CI, McNemar, Bonferroni, consensus call per cell), two-outcome full-run narrative (§3.4 here), threshold-sensitivity table — skeleton §3.4 now carries the full numbers; build the formal table from `notes/full_results_2026-08-16.md`.
- **Aug 22–23 (Sat–Sun):** Draft Intro + Related Work (use §3/§4 anchors verbatim with bib generated via the `arxiv` skill); Method section from paper1 §3–4 + harness scripts; datasets table with HF repos.
- **Aug 24 (Mon):** Discussion/Limitations incl. black-box scope, matched-denominator drops, M1 confound, RAGTurk/Halluverse caveats, single-run reproducibility.
- **Aug 25 (Tue):** **Confirm primary venue + format** (next-action from `workshop_shortlist_2026.md`): JUDGe = 4–8pp? anonymized? via OpenReview, Notif Sep 29, CR Oct 15; TAI-Eval = Aug 29 11:59pm AoE, Notif Sep 29; check **dual-submission policy** before considering both.
- **Aug 26 (Wed):** Compile to 4–8pp PDF (NeurIPS template + Tectonic; lmodern for Turkish ğ — publisher glyph pitfall from skill); run the `grounded-citations` verify gate on prose + bib (10/10 pattern from verification pass).
- **Aug 27 (Thu):** Final honesty pass — every number carries its n; no "contamination found" wording unless full-run rule met post-discount; scope-limited null phrasing everywhere.
- **Aug 28 (Fri):** Draft upload to OpenReview; reviewer-facing reproducibility block (commands from README pattern); arXiv preprint only if workshop policy allows concurrent (else workshop-only, TMLR-safe order documented in shortlist).
- **Aug 29 (Sat):** Buffer day — final AoE upload (AoE = 11:59 PM UTC-12).
- **Post-deadline:** AIDaR review Sep 14 / notif Sep 22; JUDGe+TAI-Eval notif Sep 29; JUDGe camera-ready Oct 15.

---

## 6. KEY RISKS TO THE AUG 29 PLAN

1. **Full run returned null after M1-discounting (realized 2026-08-16)** — expected and publishable (pre-registered honest negative); the Results narrative (§3.4) is now written for it, not bolted on.
2. **M1-discount adjudication is the paper's crux** — both readings (3/12 vs 0/12 at pilot; 1/12 vs 0/12 at n=200) must survive scrutiny; any reviewer challenge is answered by reporting both + the declared confound.
3. **RAGTurk cell quality** — scorer risk resolved by pre-registered fix 3 (precision-first `qa_matcher.py`); remaining caveat: low recall (0.371 measured) makes RAGTurk accuracies lower bounds — keep the lenient sensitivity variant and the floor-accuracy scope note.
4. **Venue logistics unverified:** JUDGe/TAI-Eval format, anonymization, and dual-submission policies; OpenReview account needed. Deadline dates themselves are safe (live-verified Aug 29 AoE for all three).
5. **M3 overlap may slip** — acceptable (defer with statement), but the paper couples M3 to the M1/M2 story in §9 of paper 1; keep at least the descriptive 13-gram table.
6. **Camera-ready + workshop logistics for a minor author** (JUDGe CR Oct 15; Atlanta vs Paris location; travel/virtual policy unverified) — resolve after notification, does not block Aug 29.