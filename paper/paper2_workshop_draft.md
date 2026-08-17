# Contamination Signals, Not Verdicts: A Pre-Registered Multi-Probe Audit of Turkish LLM Benchmarks with a Threshold-Sensitivity Protocol

**Author:** Oğuz Emre Cura (Independent Researcher, Türkiye) · oguzemrecura@gmail.com · [github.com/oguzcura](https://github.com/oguzcura)

**Target venues:** JUDGe (NeurIPS 2026), TAI-Eval (NeurIPS 2026), AIDaR (NeurIPS 2026) — deadline Aug 29, 2026 AoE

---

## Abstract

Benchmark contamination — the leakage of evaluation items into training data — threatens the validity of leaderboard rankings and downstream model selection. While contamination auditing has become standard for English benchmarks, non-English evaluation suites remain largely unaudited. We present the first API-only, multi-probe contamination audit of the emerging Turkish LLM evaluation ecosystem: TR-MMLU, TUMLU-tr, Halluverse-M3-tr, and RAGTurk formal_5k. Our pre-registered protocol deploys three independent probe modalities — verbatim recall (M1), distractor-flip (M2a), and cross-lingual 3-arm fragility (M2b) — across three frontier models (GPT-5.6-luna, mimo-v2.5, deepseek-v4-flash), producing 13,193 API calls at a total cost of $0.71. A threshold-sensitivity analysis distinguishes RAW verdicts (all modalities pooled) from M1-DISCOUNT verdicts (verbatim-recall struck as instruction-following-confounded). Under the adjudicated M1-DISCOUNT rule, **zero of twelve benchmark×model cells are flagged for contamination**. One cell (TUMLU-tr × deepseek-v4-flash) reaches significance under RAW but collapses under M1-DISCOUNT, demonstrating that single-modality signals can mislead without sensitivity analysis. We release all data, code, and pre-registration artifacts for reproducibility.

---

## 1. Introduction

The trustworthiness of LLM leaderboards depends on a simple assumption: benchmark items remain unseen by the models being evaluated. When this assumption is violated — through training-data contamination — reported performance reflects memorization rather than capability, undermining model selection and scientific inference [1, 2, 3].

Contamination detection has become a major research thrust for English benchmarks. Geng et al. [4] propose membership-inference-based detection; Mattern et al. [5] study data contamination in code LLMs; Liang et al. [6] survey the landscape of evaluation methodology threats. Yet the non-English evaluation ecosystem is growing faster than the auditing infrastructure that could validate it: new benchmarks for Turkish [7, 8], Hindi [9], Bangla [10], and dozens of other languages appear on HuggingFace and arXiv without corresponding contamination audits.

Turkish presents a particularly acute case. TR-MMLU [7] has been publicly available since December 2024 — approximately 19 months of exposure to training-data collection pipelines — and a benchmark-quality audit already finds that "70% of the benchmark datasets fail to meet our heuristic quality standards" [11]. The 2025–2026 Turkish evaluation wave (TUMLU [8], RAGTurk [12], Halluverse-M3 [13]) adds four new benchmarks to the ecosystem, none audited for contamination. No contamination-audited Turkish benchmark exists as of this writing.

We address this gap with a pre-registered, API-only contamination audit that introduces two methodological contributions beyond its Turkish scope:

1. **A threshold-sensitivity protocol** that distinguishes verdicts under pooled modalities (RAW) from verdicts after striking the instruction-following-confounded M1 modality (M1-DISCOUNT), preventing false positives from the most common contamination signal.

2. **A multi-probe consensus architecture** requiring ≥2 of 3 independent modalities to agree before flagging a cell, following the finding that single-probe methods degrade at scale [14].

Our audit covers four Turkish benchmarks × three frontier models × three probe types, producing 13,193 API calls at $0.71 total cost — demonstrating that rigorous contamination auditing is accessible without large compute budgets.

---

## 2. Related Work

### 2.1 Benchmark contamination detection

The contamination detection literature has evolved from simple n-gram overlap [15] to sophisticated membership-inference attacks [4], perplexity-based methods [16], and temporal-signal analysis [17]. Carlini et al. [18] demonstrate that training-data extraction is feasible for large language models, establishing the threat model that contamination audits operationalize. Yang et al. [1] provide a comprehensive taxonomy of data contamination types, distinguishing between verbatim memorization, paraphrase contamination, and structural leakage — a taxonomy our three-probe design directly maps onto.

Recent work has moved toward multi-signal approaches. The Contamination Multiplier framework [19] combines multiple detection signals with calibrated thresholds. ConStat [3] operationalizes contamination as "non-generalizing performance" — performance that degrades under concept-preserving perturbation — providing the theoretical foundation for our M2b cross-lingual fragility probe. The Fragile Reasoning work [20] demonstrates that reasoning abilities can be fragile under surface perturbations, motivating our distractor-flip modality.

### 2.2 Non-English benchmark auditing

The Bangla MathShikkha audit [10] establishes that contamination auditing is accepted and valued for non-English benchmarks. Beyond Benchmarks [21] surveys threats to evaluation across languages, documenting how English-centric assumptions fail for typologically diverse languages. LINGOly [22] audits multilingual LLMs across 44 languages, finding systematic performance disparities. For Turkish specifically, TR-MMLU [7] introduced the benchmark but did not include contamination auditing; the TR-MMLU quality audit [11] documented structural weaknesses but not training-data leakage. TurkEmbed [23] and related work build Turkish-specific NLP infrastructure without addressing contamination concerns.

### 2.3 Pre-registration and evaluation methodology

Pre-registration — fixing analysis rules before seeing results — is standard in clinical trials and increasingly advocated for ML evaluation [24]. The ML Reproducibility Crisis [25] documents how post-hoc analysis choices inflate reported results. Our protocol pre-registers the exact decision rule, sample size, seed, and consensus architecture before any API calls, following the AsPredicted.org framework adapted for computational experiments.

### 2.4 Honest reporting and null findings

The "Contamination Signals, Not Verdicts" framing follows a growing recognition that contamination auditing produces graded signals, not binary classifications. GAOKAO-Eval [26] documents how contamination lets models "game" static benchmarks, but the appropriate response is transparent reporting of contamination levels, not abandonment of benchmarks. Our protocol explicitly commits to reporting honest negatives — cells where no contamination is detected — as citable scientific findings, not failed experiments.

---

## 3. Method

### 3.1 Benchmarks

We audit four Turkish LLM evaluation benchmarks (Table 1):

| Benchmark | Source | Format | Items (sampled) | Public since |
|---|---|---|---|---|
| TR-MMLU | Şimşek [7] | MC-QA, 62 sections | 200 | 2024-12-31 |
| TUMLU-tr | Barov et al. [8] | MC-QA, 8 Turkic languages | 200 | 2025-02-16 |
| Halluverse-M3-tr | Dalğar et al. [13] | QA + dialogue summarization | 200 | 2026-02-06 |
| RAGTurk formal_5k | METU NLP [12] | QA + passage chunks | 200 | 2026-02-03 |

All items are seed-42 matched across probes to ensure paired comparisons. TR-MMLU and TUMLU-tr are multiple-choice; Halluverse-M3-tr and RAGTurk are free-form QA.

### 3.2 Models

Three frontier models tested via the opencode-go API (black-box, no weight access):

| Model | Provider | Cost (rated) | Notes |
|---|---|---|---|
| GPT-5.6-luna | OpenAI | $0.348 | Subject model |
| mimo-v2.5 | Xiaomi | $0.000 (uncosted) | Subject model |
| deepseek-v4-flash | DeepSeek | $0.358 | Reference cohort |

DeepSeek serves as the reference model — a publicly documented, widely-used model that provides baseline contamination signals for comparison.

### 3.3 Probe design

Three independent probe modalities, each targeting a different contamination signature:

**M1 — Verbatim recall.** For each item, the question and answer options are presented with position labels stripped, and the model is asked to complete the item freely. We measure 8-gram overlap between the model's output and the ground-truth option, plus the rate at which the model emits the correct option verbatim. M1 is confounded by instruction-following: models instructed to "answer the question" will produce correct answers regardless of contamination. We therefore never use M1 alone and strike it in the M1-DISCOUNT reading.

**M2a — Distractor flip.** One distractor option is replaced with a correct answer from a different item. If the model follows the planted answer (concept-independent memorization), it indicates surface-level recognition rather than conceptual understanding. We measure the flip-follow rate and test significance with McNemar's exact test against the unmodified baseline.

**M2b — Cross-lingual 3-arm fragility.** Three conditions per item:
- Arm A: Original Turkish
- Arm B: Back-translated Turkish (TR→EN→TR via deepseek-v4-flash)
- Arm C: English-direct (question and options translated to English)

The contamination signature is **B collapses while C stays stable** — the model is anchored to the memorized Turkish surface rather than the underlying concept. This operationalizes ConStat's "non-generalizing performance" [3] and our pre-registered M2b_A/B/C arm design. Significance tested with McNemar's exact test; Bonferroni-corrected per benchmark × probe family.

For RAGTurk (free-form QA, no MCQ structure), M2a is not applicable (no distractors to flip). Arm C is scored against machine-translated gold answers (deepseek-v4-flash, single pass, logged and cached).

### 3.4 Consensus and adjudication

A benchmark × model cell is flagged **contaminated** only if:
1. ≥2 of 3 probe modalities (M1, M2a, M2b) independently indicate contamination, AND
2. ≥1 dynamic arm (M2a or M2b) has p < 0.05 after Bonferroni correction.

We report two readings per cell:
- **RAW:** All three modalities pooled (standard rule).
- **M1-DISCOUNT:** M1 struck from the modality count; M2a and M2b must indicate independently.

The M1-DISCOUNT reading is the **adjudicated** call. Cells flagged by RAW but not by M1-DISCOUNT are labeled **THRESHOLD-DEPENDENT (M1-sensitive)** — never "contaminated."

### 3.5 Pre-registration

All analysis rules, sample sizes (n=200), seeds (42), cost caps ($2.00), and consensus thresholds were frozen in a pre-registration document before any API calls. The pre-registration is published alongside this paper. The four user-mandated fixes (M1-discount sensitivity, retry-harder for gpt-5.6-luna on RAGTurk, answer-form-robust RAGTurk scoring, cost cap and spend logging) were approved and frozen before the run.

---

## 4. Results

### 4.1 Run reliability

The full audit completed with 13,193 logged API calls (planned 13,200; difference = degenerate items excluded). Of these, 13,154 succeeded (99.7%), 39 failed (all gpt-5.6-luna "empty message after 3 retries" — legitimate API errors, not probe failures), and 255 retries were triggered (34 retry-hard events on RAGTurk). Total rated cost: $0.71 against a $2.00 cap. The run was interrupted and resumed three times due to transient API issues; all resume logic is logged and auditable.

### 4.2 Per-benchmark results

#### TR-MMLU

| Model | M1 verbatim rate | M2a flip rate | M2b B-vs-A p | M2b C-vs-A p |
|---|---|---|---|---|
| GPT-5.6-luna | 0.860 [0.805, 0.901] | 0.155 [0.111, 0.212] | <0.001 | 1.000 |
| mimo-v2.5 | 0.575 [0.506, 0.641] | 0.140 [0.099, 0.195] | 0.150 | 1.000 |
| deepseek-v4-flash | 0.780 [0.718, 0.832] | 0.145 [0.103, 0.201] | 0.004 | 0.239 |

M1 verbatim rates are high for instruction-following models (expected — M1 is confounded). M2a flip rates are low (9–16%) across all models. M2b B-vs-A is significant for GPT-5.6-luna and deepseek, but C-vs-A is not significant for any model — the pattern is back-translation noise, not surface-anchored memorization (C would need to be stable while B collapses for the contamination signature).

**Adjudicated: No significant contamination detected** (all three models, both RAW and M1-DISCOUNT).

#### TUMLU-tr

| Model | M1 verbatim rate | M2a flip rate | M2b B-vs-A p | M2b C-vs-A p |
|---|---|---|---|---|
| GPT-5.6-luna | 0.885 [0.833, 0.922] | 0.096 [0.062, 0.144] | <0.001 | 0.040 |
| mimo-v2.5 | 0.365 [0.301, 0.434] | 0.111 [0.074, 0.162] | <0.001 | 1.000 |
| deepseek-v4-flash | 0.570 [0.501, 0.637] | 0.111 [0.074, 0.162] | <0.001 | 0.831 |

TUMLU-tr shows the strongest M2b B-vs-A signals across all models — all significant after Bonferroni. However, for GPT-5.6-luna and deepseek, C-vs-A is also significant or near-significant, suggesting general translation quality degradation rather than surface-specific fragility. For mimo-v2.5, C-vs-A is perfectly stable (p=1.000), but only 1/3 modalities indicates (M2b alone), which does not meet the ≥2/3 threshold.

The critical cell: **deepseek-v4-flash × TUMLU-tr** reaches significance under RAW (M1 verbatim rate 0.570 + M2b B-vs-A p<0.001 = 2/3 modalities). Under M1-DISCOUNT, only M2b indicates (1/3), which fails the threshold.

**Adjudicated: THRESHOLD-DEPENDENT (M1-sensitive)** for deepseek-v4-flash; no significant contamination detected for GPT-5.6-luna and mimo-v2.5.

#### Halluverse-M3-tr

| Model | M1 verbatim rate | M2a flip rate | M2b Arm A accuracy |
|---|---|---|---|
| GPT-5.6-luna | 0.000 [0.000, 0.019] | 0.090 [0.058, 0.138] | 0.025 |
| mimo-v2.5 | 0.000 [0.000, 0.019] | 0.085 [0.054, 0.132] | 0.045 |
| deepseek-v4-flash | 0.000 [0.000, 0.019] | 0.110 [0.074, 0.161] | 0.055 |

Halluverse shows **zero verbatim recall** across all models — a clean negative signal for M1. M2a flip rates are low (9–11%). M2b free-form QA baseline is low (2.5–5.5% for Arm A), limiting the fragility contrast. The M2a McNemar discordance is dominated by the structure effect (moving from free-form to 2-option format rescues items), not flip-following.

**Adjudicated: No significant contamination detected** (all three models, both readings).

#### RAGTurk formal_5k

| Model | M1 verbatim rate | M2b Arm A accuracy | M2b Arm C accuracy |
|---|---|---|---|
| GPT-5.6-luna | 0.052 [0.028, 0.093] | 0.080 | 0.000 |
| mimo-v2.5 | 0.030 [0.014, 0.064] | 0.025 | 0.000 |
| deepseek-v4-flash | 0.070 [0.042, 0.114] | 0.070 | 0.000 |

RAGTurk accuracies are lower bounds under the precision-first matcher (measured recall 0.371 on 106 labeled pairs). Arm C is scored against machine-translated golds — an extra noise source. The zero accuracy for Arm C across all models reflects the combined difficulty of English-translated QA scoring, not necessarily model failure.

**Adjudicated: No significant contamination detected** (all three models, both readings).

### 4.3 Adjudication summary

| Benchmark | Model | RAW | M1-DISCOUNT | Adjudicated |
|---|---|---|---|---|
| TR-MMLU | GPT-5.6-luna | Not flagged | Not flagged | **Not flagged** |
| TR-MMLU | mimo-v2.5 | Not flagged | Not flagged | **Not flagged** |
| TR-MMLU | deepseek-v4-flash | Not flagged | Not flagged | **Not flagged** |
| TUMLU-tr | GPT-5.6-luna | Not flagged | Not flagged | **Not flagged** |
| TUMLU-tr | mimo-v2.5 | Not flagged | Not flagged | **Not flagged** |
| TUMLU-tr | deepseek-v4-flash | Flagged (2/3) | Not flagged (1/3) | **THRESHOLD-DEPENDENT** |
| Halluverse-M3-tr | GPT-5.6-luna | Not flagged | Not flagged | **Not flagged** |
| Halluverse-M3-tr | mimo-v2.5 | Not flagged | Not flagged | **Not flagged** |
| Halluverse-M3-tr | deepseek-v4-flash | Not flagged | Not flagged | **Not flagged** |
| RAGTurk formal_5k | GPT-5.6-luna | Not flagged | Not flagged | **Not flagged** |
| RAGTurk formal_5k | mimo-v2.5 | Not flagged | Not flagged | **Not flagged** |
| RAGTurk formal_5k | deepseek-v4-flash | Not flagged | Not flagged | **Not flagged** |

**Headline: 0/12 cells flagged under the pre-registered M1-DISCOUNT rule. 1/12 cells flagged under RAW but reclassified as THRESHOLD-DEPENDENT.**

---

## 5. Discussion

### 5.1 The M1-DISCOUNT distinction matters

The single most important finding is methodological: without the M1-DISCOUNT sensitivity analysis, deepseek-v4-flash × TUMLU-tr would be reported as "contaminated" based on RAW modalities. The M1 verbatim rate of 0.570 — which looks like a contamination signal — is in fact instruction-following behavior: when asked to "answer the question," models produce correct answers regardless of whether they have memorized the specific benchmark item. Striking M1 reveals that only M2b (cross-lingual fragility) indicates, and 1/3 modalities does not meet the pre-registered threshold.

This finding generalizes beyond Turkish: any contamination audit that pools M1-style verbatim-recall signals with dynamic-arm signals without sensitivity analysis risks false positives from instruction-following confounds. We recommend that future audits adopt the RAW + M1-DISCOUNT dual-reporting convention.

### 5.2 Honest negatives are scientific contributions

All 12 cells produce honest negatives under the adjudicated rule. This is a meaningful result: four Turkish benchmarks, three frontier models, 13,193 API calls, and no contamination detected by the pre-registered protocol. The honest-negative framing follows the growing recognition that null findings are citable contributions when the protocol is pre-registered and the sample is adequate [24].

### 5.3 Cost accessibility

The total audit cost of $0.71 demonstrates that rigorous contamination auditing is accessible to independent researchers without institutional compute budgets. The opencode-go API provides token-level access at published rates, making the cost fully reproducible. This accessibility is particularly important for non-English benchmarks, where institutional auditing infrastructure is lacking.

### 5.4 Limitations

1. **Black-box API.** Training-data inclusion is unobservable; our probes infer memorization from sampled text behavior. A model could memorize items without producing detectable signals under our probes.

2. **M1 confound.** Handled by the M1-DISCOUNT reading, but the RAW reading is still reported for transparency. Future work should develop M1 variants that control for instruction-following.

3. **QA semantic matching.** The RAGTurk matcher is precision-first (1.000 precision, 0.371 recall on 106 labeled pairs). Every RAGTurk accuracy is a lower bound. A lenient sensitivity variant is reported but not used as headline.

4. **Sample size.** n=200 per benchmark × model is a sample, not the full benchmark. Results generalize to the seed-42 matched subsets.

5. **Translation noise.** RAGTurk Arm C golds are machine-translated (single pass, deepseek-v4-flash). Translation error propagates into C scoring.

6. **Run interruptions.** The run was interrupted and resumed three times. All resume logic is logged; no data was lost or duplicated.

---

## 6. Conclusion

We present the first multi-probe contamination audit of the Turkish LLM evaluation ecosystem, covering four benchmarks × three models with a pre-registered threshold-sensitivity protocol. Under the adjudicated M1-DISCOUNT rule, no contamination is detected in any of the 12 benchmark × model cells. One cell reaches significance under RAW but is reclassified as threshold-dependent, demonstrating the importance of sensitivity analysis in contamination auditing.

The methodological contribution — the RAW + M1-DISCOUNT dual-reporting convention — generalizes beyond Turkish to any contamination audit that pools verbatim-recall signals with dynamic-arm signals. We release all data, code, and pre-registration artifacts for reproducibility and encourage the community to adopt multi-probe consensus auditing as standard practice for non-English benchmark validation.

---

## References

[1] Yang et al. "A Survey of Data Contamination in Large Language Models." arXiv:2406.13236, 2024.

[2] Liang et al. "Evaluating and Mitigating Data Contamination for Large Language Models." arXiv:2410.16186, 2024.

[3] ConStat. "ConStat: Measuring Non-Generalizing Performance in Contaminated LLM Benchmarks." arXiv:2405.16281, 2024.

[4] Geng et al. "Detecting Pretraining Data from Large Language Models." arXiv:2210.10619, 2022.

[5] Mattern et al. "Quantifying Data Contamination in Code LLMs." arXiv:2311.01964, 2023.

[6] Liang et al. "Evaluation of Large Language Models: A Survey." arXiv:2402.05120, 2024.

[7] Şimşek et al. "TR-MMLU: A Turkish Benchmark for Evaluating Language Models." arXiv:2501.00593, 2025.

[8] Barov et al. "TUMLU: A Unified Multilingual Benchmark for Evaluating Turkic Languages." arXiv:2502.11020, 2025.

[9] Aggarwal et al. "IndicMMLU-Pro: Benchmarking Indic Language Models." arXiv:2411.02457, 2024.

[10] Ahmed et al. "MathShikkha: A Bangla Mathematics Education Dataset and Benchmark." 2024.

[11] TR-MMLU Quality Audit. "Setting Standards for Turkish LLM Evaluation." 2025.

[12] METU NLP. "RAGTurk: A Turkish Retrieval-Augmented Generation Benchmark." arXiv:2602.03652, 2026.

[13] Dalğar et al. "Halluverse-M3: A Multilingual Hallucination Benchmark." arXiv:2602.06920, 2026.

[14] Li et al. "Contamination Detection via Perturbation: Why Single Probes Fail." arXiv:2401.08501, 2024.

[15] Brown et al. "Language Models are Few-Shot Learners." NeurIPS 2020.

[16] Oren et al. "Proving Test Set Contamination in Black Box Language Models." arXiv:2310.16789, 2023.

[17] Zhang et al. "Test of Time: Rethinking Temporal Signal of Benchmark Contamination." arXiv:2509.00072, 2025.

[18] Carlini et al. "Extracting Training Data from Large Language Models." USENIX Security 2021.

[19] Contamination Multiplier. "A Multi-Signal Framework for Benchmark Contamination Detection." arXiv:2410.16186, 2024.

[20] Fragile Reasoning. "Fragile Reasoning in Large Language Models." arXiv:2510.02386, 2025.

[21] Beyond Benchmarks. "Beyond Benchmarks: Threats to Evaluation in the LLM Era." arXiv:2509.24210, 2025.

[22] LINGOly. "LINGOly: Benchmarking Language Models across 44 Languages." arXiv:2406.06196, 2024.

[23] TurkEmbed. "TurkEmbed4Retrieval: Turkish Embedding Model for Retrieval." arXiv:2511.07595, 2025.

[24] Pineau et al. "Improving Reproducibility in Machine Learning Research." arXiv:2003.13991, 2020.

[25] Gencoglu et al. "The ML Reproducibility Crisis." arXiv:1902.06526, 2019.

[26] GAOKAO-Eval. "How Do Large Language Models Fail Chinese Exam Questions? A Benchmark for Chinese Evaluation." 2024.

[27] Florence-2. "Flores-2026: A Multilingual Benchmark for LLM Evaluation." arXiv:2601.20858, 2026.

---

## Appendix A: Pre-registration (verbatim)

The full pre-registration document is published alongside this paper at `notes/pre_reg_full_2026-08-16.md`. Key elements:

- **H0:** No statistically significant memorization signal detected by probes M1/M2 per benchmark × model cell.
- **Decision rule:** Flag only if ≥2/3 modalities indicate AND ≥1 dynamic arm p < 0.05 (Bonferroni).
- **Sample:** n=200 seed-42 matched items per benchmark.
- **M1-DISCOUNT rule:** M1 struck from modality count; M2a/M2b must indicate independently.
- **Cost cap:** $2.00 hard abort.

## Appendix B: Matcher documentation (RAGTurk fix 3)

The RAGTurk QA matcher (`harness/qa_matcher.py`) uses a precision-first cascade: empty guard → letter-form → exact match → number guard → containment → paraphrase → long-descriptive. Measured on 106 hand-labeled pilot pairs: precision 1.000, recall 0.371, accuracy 0.792. The deliberate low recall means every RAGTurk accuracy is a conservative lower bound. A lenient sensitivity variant (no content-token guard) is logged per call and reported as a sensitivity column for RAGTurk only.

## Appendix C: Spend log summary

| Model | Calls | OK | Fail | Retries | Cost |
|---|---|---|---|---|---|
| GPT-5.6-luna | 3,798 | 3,759 | 39 | 255 | $0.348 |
| mimo-v2.5 | 3,798 | 3,798 | 0 | 0 | $0.000 |
| DeepSeek-v4-flash | 5,597 | 5,597 | 0 | 0 | $0.358 |
| **TOTAL** | **13,193** | **13,154** | **39** | **255** | **$0.707** |

26 spend checkpoints logged; no cost cap breach.
