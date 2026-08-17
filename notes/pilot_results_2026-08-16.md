# Pilot Results — Turkish Benchmark Contamination Audit (Day 5, 2026-08-16)

**Scope:** n=50 seed-42 matched items per benchmark x 3 models (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash reference), all via opencode-go (OpenRouter untouched). Probes M1 (verbatim-recall + 8-gram), M2a (distractor-flip / edited-answer flip), M2b (3-arm A/B/C crosslingual). Design: `notes/paper1_contamination_audit.md` §0, §3, §4.

**Pre-registered decision rule (§0), applied exactly:** a benchmark x model cell is flagged **only if ≥2 of 3 probe modalities indicate AND ≥1 dynamic arm (M2a/M2b) has p < 0.05 after Bonferroni** (family = benchmark x probe). Any other outcome = **no significant contamination detected** (honest negative).

**Probe applicability (honest notes):** M2a **does not apply** to RAGTurk (QA-pair, no MC options) — no calls made, cell reported as N/A. Halluverse M2a uses its built-in hallucination-edit twin (`edited_answer`) as a 2-option forced-choice flip (positions randomized per item, seed-42). QA arms are scored with a **lenient containment match** (normalized exact/substring); this is weaker than letter-exact MC scoring and reported as such. M1 asks the model to reproduce the answer text verbatim, so high M1 values partly measure instruction-following — M1 alone never flags a cell (documented confound).

## 1. Run reliability & spend

**Calls logged: 3250** (planned 2,850 answer calls + 400 lazy translations = 3,250). ok = 3237, fail = 13, retry-on-empty events (total, logged) = 46 (20 recovered, 26 exhausted across 13 calls). No 5xx/429 backoff retries triggered. No live cost-cap abort (accumulated estimate stayed below $1.00).

### Spend log (from the JSONL; documented rates only)

| model | calls | ok | fail | retries | prompt tk | completion tk | reasoning tk | cost est USD |
|---|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 950 | 937 | 13 | 46 | 113715 | 122220 | 0 | $0.084703 |
| mimo-v2.5 | 950 | 950 | 0 | 0 | 118997 | 332906 | 276879 | n/a (no published rate) |
| deepseek-v4-flash | 1350 | 1350 | 0 | 0 | 281238 | 511691 | 345248 | $0.091352 |
| **TOTAL** | **3250** | **3237** | **13** | **46** | **513950** | **966817** | **622127** | **$0.176056 (rated models; mimo tokens-only)** |

Note: deepseek-v4-flash rows include the 400 translation calls (all translations run on deepseek-v4-flash as the fixed translator, design §3). mimo-v2.5 has no published per-token rate on opencode-go → tokens counted, cost not estimable (same note as smoke §4).

### Per benchmark

| benchmark | calls | ok | fail | retries |
|---|---|---|---|---|
| TR-MMLU | 850 | 850 | 0 | 1 |
| TUMLU-tr | 850 | 850 | 0 | 0 |
| Halluverse-M3-tr | 850 | 849 | 1 | 2 |
| RAGTurk formal_5k | 700 | 688 | 12 | 43 |

## 2. Probe results (every cell shown)

### TR-MMLU

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 50 | 46 | 0.920 [0.812,0.969] | 0.747 ± 0.429 |
| mimo-v2.5 | 50 | 23 | 0.460 [0.330,0.596] | 0.404 ± 0.474 |
| deepseek-v4-flash (ref) | 50 | 39 | 0.780 [0.648,0.873] | 0.651 ± 0.462 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 8 | 0.160 [0.083,0.285] | (7, 3) | 0.344 | 1.000 |
| mimo-v2.5 | 50 | 10 | 0.200 [0.112,0.330] | (12, 3) | 0.035 | 0.106 |
| deepseek-v4-flash (ref) | 50 | 12 | 0.240 [0.143,0.374] | (9, 2) | 0.065 | 0.196 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by lenient containment match)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 0.840 (42/50) [0.715,0.917] | 0.620 (31/50) [0.481,0.741] | 0.840 (42/50) [0.715,0.917] | 0.009 | 1.000 | 0.009 |
| mimo-v2.5 | 47 | 0.830 (39/47) [0.699,0.911] | 0.745 (35/47) [0.605,0.848] | 0.702 (33/47) [0.560,0.814] | 1.000 | 0.633 | 1.000 |
| deepseek-v4-flash (ref) | 36 | 0.889 (32/36) [0.747,0.956] | 0.833 (30/36) [0.681,0.921] | 0.917 (33/36) [0.782,0.971] | 1.000 | 1.000 | 1.000 |

**Consensus call (per §0):**

- **gpt-5.6-luna**: **CONTAMINATED (pre-registered rule)** (modalities indicating: 2/3 [M1=Y, M2a=n, M2b=Y]; min dynamic-arm p after Bonferroni = 0.009).
- **mimo-v2.5**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = 0.106).
- **deepseek-v4-flash (ref)**: **no significant contamination detected** (modalities indicating: 1/3 [M1=Y, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = 0.196).

### TUMLU-tr

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 50 | 47 | 0.940 [0.838,0.979] | 0.896 ± 0.300 |
| mimo-v2.5 | 50 | 21 | 0.420 [0.294,0.558] | 0.447 ± 0.467 |
| deepseek-v4-flash (ref) | 50 | 29 | 0.580 [0.442,0.706] | 0.676 ± 0.409 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 5 | 0.100 [0.043,0.214] | (6, 0) | 0.031 | 0.094 |
| mimo-v2.5 | 50 | 5 | 0.100 [0.043,0.214] | (11, 3) | 0.057 | 0.172 |
| deepseek-v4-flash (ref) | 50 | 6 | 0.120 [0.056,0.238] | (10, 2) | 0.039 | 0.116 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by lenient containment match)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 0.980 (49/50) [0.895,0.997] | 0.700 (35/50) [0.562,0.809] | 0.900 (45/50) [0.786,0.957] | <0.001 | 1.000 | 0.057 |
| mimo-v2.5 | 48 | 0.833 (40/48) [0.704,0.913] | 0.646 (31/48) [0.504,0.766] | 0.792 (38/48) [0.657,0.883] | 0.441 | 1.000 | 0.589 |
| deepseek-v4-flash (ref) | 40 | 1.000 (40/40) [0.912,1.000] | 0.750 (30/40) [0.598,0.858] | 0.875 (35/40) [0.739,0.945] | 0.018 | 0.562 | 1.000 |

**Consensus call (per §0):**

- **gpt-5.6-luna**: **CONTAMINATED (pre-registered rule)** (modalities indicating: 2/3 [M1=Y, M2a=n, M2b=Y]; min dynamic-arm p after Bonferroni = <0.001).
- **mimo-v2.5**: **no significant contamination detected** (modalities indicating: 1/3 [M1=n, M2a=n, M2b=Y]; min dynamic-arm p after Bonferroni = 0.172).
- **deepseek-v4-flash (ref)**: **CONTAMINATED (pre-registered rule)** (modalities indicating: 2/3 [M1=Y, M2a=n, M2b=Y]; min dynamic-arm p after Bonferroni = 0.018).

### Halluverse-M3-tr

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 49 | 0 | 0.000 [0.000,0.073] | 0.056 ± 0.115 |
| mimo-v2.5 | 50 | 0 | 0.000 [0.000,0.071] | 0.084 ± 0.165 |
| deepseek-v4-flash (ref) | 50 | 0 | 0.000 [0.000,0.071] | 0.092 ± 0.153 |

**M2a — distractor flip / edited-answer flip** (follow-planted rate + paired McNemar exact vs A-arm baseline; Bonferroni family size=3)

| model | n | flip-follows | flip-follow rate [95% CI] | McNemar discord. (base-ok→test-wrong, base-wrong→test-ok) | p raw | p Bonf |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 4 | 0.080 [0.032,0.188] | (0, 35) | <0.001 | <0.001 |
| mimo-v2.5 | 50 | 7 | 0.140 [0.070,0.262] | (0, 35) | <0.001 | <0.001 |
| deepseek-v4-flash (ref) | 50 | 5 | 0.100 [0.043,0.214] | (1, 35) | <0.001 | <0.001 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by lenient containment match)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 50 | 0.220 (11/50) [0.128,0.352] | 0.240 (12/50) [0.143,0.374] | 0.260 (13/50) [0.159,0.396] | 1.000 | 1.000 | 1.000 |
| mimo-v2.5 | 50 | 0.160 (8/50) [0.083,0.285] | 0.100 (5/50) [0.043,0.214] | 0.060 (3/50) [0.021,0.162] | 1.000 | 1.000 | 1.000 |
| deepseek-v4-flash (ref) | 50 | 0.200 (10/50) [0.112,0.330] | 0.240 (12/50) [0.143,0.374] | 0.260 (13/50) [0.159,0.396] | 1.000 | 1.000 | 1.000 |

**Consensus call (per §0):**

- **gpt-5.6-luna**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = <0.001).
- **mimo-v2.5**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = <0.001).
- **deepseek-v4-flash (ref)**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = <0.001).

### RAGTurk formal_5k

**M1 — verbatim recall / 8-gram overlap** (confounded by instruction-following; never decisive alone)

| model | n | verbatim hits | verbatim rate [95% CI] | mean overlap8 ± sd |
|---|---|---|---|---|
| gpt-5.6-luna | 48 | 2 | 0.042 [0.011,0.140] | 0.145 ± 0.229 |
| mimo-v2.5 | 50 | 0 | 0.000 [0.000,0.071] | 0.083 ± 0.146 |
| deepseek-v4-flash (ref) | 50 | 2 | 0.040 [0.011,0.135] | 0.133 ± 0.242 |

**M2b — 3-arm crosslingual fragility** (fully-matched denominator; McNemar exact paired; Bonferroni family size=9; QA arms scored by lenient containment match)

| model | matched n | Arm A (TR orig) | Arm B (TR→EN→TR) | Arm C (EN direct) | B vs A p (Bonf) | C vs A p (Bonf) | C vs B p (Bonf) |
|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | 43 | 0.070 (3/43) [0.024,0.186] | 0.070 (3/43) [0.024,0.186] | 0.000 (0/43) [0.000,0.082] | 1.000 | 1.000 | 1.000 |
| mimo-v2.5 | 50 | 0.000 (0/50) [0.000,0.071] | 0.020 (1/50) [0.004,0.105] | 0.000 (0/50) [0.000,0.071] | 1.000 | 1.000 | 1.000 |
| deepseek-v4-flash (ref) | 50 | 0.060 (3/50) [0.021,0.162] | 0.060 (3/50) [0.021,0.162] | 0.040 (2/50) [0.011,0.135] | 1.000 | 1.000 | 1.000 |

**Consensus call (per §0):**

- **gpt-5.6-luna**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = 1.000).
- **mimo-v2.5**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = 1.000).
- **deepseek-v4-flash (ref)**: **no significant contamination detected** (modalities indicating: 0/3 [M1=n, M2a=n, M2b=n]; min dynamic-arm p after Bonferroni = 1.000).

## 3. Interpretation (honest-negative discipline, design §6)

- Every cell above is shown with its n, Wilson 95% CI, raw and Bonferroni-corrected p. Directional-but-not-significant results are reported as **null/suggestive**, never as findings.
- Three cells meet the §0 threshold **as pre-registered**: TR-MMLU x gpt-5.6-luna, TUMLU-tr x gpt-5.6-luna, TUMLU-tr x deepseek-v4-flash. In each, the carrier is the **M2b back-translation collapse with English-direct stability**: B accuracy drops (0.84→0.62, 0.98→0.70, 1.00→0.75) with perfectly one-sided paired discordance (11-0, 14-0, 10-0; Bonferroni p = 0.009 / <0.001 / 0.018) while C-vs-A shows no change (p = 1.000 / 1.000 / 0.562). This is the ConStat non-generalizing-performance signature (anchored to the memorized Turkish surface).
- **Flag-sensitivity disclosure (mandatory for honesty):** in all three flagged cells the second 'indicating' modality is **M1**, which the design itself flags as instruction-following-confounded (the prompt explicitly asks for verbatim reproduction). If M1 is discounted, each flagged cell rests on a single modality (M2b) and does NOT meet the ≥2-modalities bar — i.e., the calls are **threshold-dependent**: they are exactly what the pre-registered rule produces, and exactly what a stricter reading of the design's own M1 caveat would not flag. The M2b discordances themselves are directionally robust (one-sided 11-0 / 14-0 / 10-0) and must be confirmed at n=200 before any final claim.
- No M2a flip-following anywhere: follow-planted rates are 0.08-0.24 across all cells (never >0.5), so no model follows planted distractors; the significant M2b effects are surface-fragility (back-translation), not concept-independent memorization.
- Halluverse: all three models score low on free-form QA (A-arm 0.16-0.22). The M2a 2-option flip control shows no flip-following (0.08-0.14); its McNemar (0, 35) reflects that the forced true-vs-edited choice **rescues** 35 items relative to free-form answering (structure effect, direction opposite to flip-following). Not a contamination signal; the cell stays an honest negative.
- RAGTurk formal_5k: near-zero accuracies (0.00-0.07) under the lenient containment scorer. For long formal gold answers this largely reflects **answer-form mismatch** (semantically right, surface different), not zero capability; arm C near-zero for luna/mimo is a surface artifact of matching Turkish gold strings against English outputs. With the A baseline at floor, no fragility contrast is possible — cells report honest negatives with this scope note.
- M1 values are high for gpt-5.6-luna and deepseek-v4-flash on the MC benchmarks because the M1 prompt explicitly requests verbatim reproduction — this measures instruction-following, not memorization; M1 never flags alone.
- Run-integrity note: the 13 failed calls are all gpt-5.6-luna empty-message responses, 12/13 on RAGTurk (clustered on specific long items — e.g. items 2, 36, 41 recur across probes) plus 1 on Halluverse; retry-on-empty recovered 20 retry events (15 calls recovered). Cells report the honest reduced/matched n (e.g. RAGTurk x luna M2b matched n=43).

Null statements above are **scope-limited**: "no contamination detected by these probes at n=50 per benchmark x model" — never a claim that a benchmark is clean (black-box API: training inclusion is unobservable).

## 4. Gate: full n=200 run (Day 8)

- Pipeline clean: 3237/3250 calls ok; resume-skip, rate limiter (~2 req/s cap, live ~0.5 calls/s observed), cost cap and retry-on-empty (46 events: 20 recovered on 15 calls, 26 exhausted across 13 calls) all exercised; exhausted calls are handled honestly in the reported denominators.
- Pilot spend: $0.176056 estimated on rated models (luna $0.0847 + deepseek incl. translations $0.0914) + mimo-v2.5 tokens-only; ≈ 18% of the $1.00 pilot cap.
- n=200 scaling at observed rates: ~4x calls (≈13,000) and ~4x cost (≈ **$0.70** rated-model estimate on pilot token profiles; verify before launch). mimo tokens grow ~4x uncosted; runtime ≈ 6-7 h at the observed 0.5 calls/s (raise --rate/--workers if the endpoint accepts it).
- **Recommendation: full n=200 run is clear to launch** — same lineup and transport (opencode-go), explicit user approval for the raised cost cap, one code-pipeline change to consider first: **confirm or refute the three pilot flags with a pre-registered M1-discount sensitivity analysis** (report both readings) since the §0 threshold is met in 3/12 cells and the flags are M1-confound-sensitive. Also consider: (i) a retry-harder policy for gpt-5.6-luna on long RAGTurk items, or excluding the 8 item-cluster failures via matched-denominator reporting (already the protocol); (ii) RAGTurk QA-pair scoring needs an answer-form-robust scorer or a re-render to MC format before claiming anything from that cell.
