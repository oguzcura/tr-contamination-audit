# Paper 1 — Turkish Benchmark Contamination Audit (trmlu-audit v2)
**Working title (workshop length):** *"How Contaminated Are Turkish LLM Benchmarks? A Multi-Probe Memorization Audit of TR-MMLU and the 2025–2026 Turkish Evaluation Wave"*
**Design date:** 2026-08-16 · **Status:** full design, pre-execution · **Venue target:** ACL/COLING workshop (or EMNLP workshop) — see §8.
**Author:** oguzc · **Hardware:** none (pure API black-box). All content below was verified live this session (arXiv API + Hugging Face API); every arXiv ID resolves and every dataset repo is confirmed present.

---

## 0. PRE-REGISTRATION (frozen 2026-08-16, Day 1)
- **H0 (null):** no statistically significant memorization signal detected by probes M1/M2 per benchmark×model cell.
- **H1a/H1b:** verbatim-recognition above chance; surface-fragility (B-drop with C-stable) — per §1.3.
- **Decision rule (pre-registered, not negotiable post-hoc):** a benchmark×model cell is flagged "contaminated" **only if ≥2 of 3 probe modalities indicate** AND **≥1 dynamic arm (M2a/M2b) has p < 0.05 after Bonferroni**. Any other outcome = "no significant contamination detected" (honest negative), reported with every cell shown.
- **Sample/seed:** fixed seed 42, n=200 matched target per benchmark, fully-matched denominator reported honestly (existing 175/200 pattern).
- **Models:** 3 independent families (Anthropic / Google / OpenAI) + DeepSeek v4 Flash reference cohort, pure black-box via OpenRouter.
- **Cost gates:** pilot n=50 (Day 5) and full run n=200 (Day 8) require explicit user approval AFTER smoke tests pass. Smoke tests (n≤20) pre-approved 2026-08-16. **Pilot PRE-APPROVED by user 2026-08-16** (conditional: smoke tests pass + lineup in §5 confirmed).

---

## 1. Research question & hypotheses

### 1.1 Central research question (single-sentence)
> To what extent do frontier, API-only black-box LLMs memorize and reproduce items from Turkish evaluation benchmarks — TR-MMLU and the new 2025–2026 Turkish benchmark wave (TUMLU, RAGTurk, Halluverse-M³ Turkish, TAS) — as measured by three independent probe modalities (verbatim recall, option perturbation, corpus overlap)?

### 1.2 Motivation (why this is publishable now)
Benchmark trust is the dominant evaluation-research theme of this window [4][9]. Turkish is exactly the low-resource case where the concern bites hardest: TR-MMLU has been public since 2024-12-31 [1] — a full ~19 months of exposure to training-data collection, and a benchmark-quality audit already finds "70% of the benchmark datasets fail to meet our heuristic quality standards" for Turkish [3]. The first (and so far only) contamination-audited *Turkish* benchmark does not exist; the Bangla precedent (MathShikkha) shows the audit genre is accepted and rewarded [8]. GAOKAO-Eval documents how leakage lets models "game" static benchmarks [27]. This is the Turkish first-of-kind audit.

### 1.3 Hypotheses — honest framing to protect against confirmation bias
**H0 (null / default):** For the sampled items of each benchmark×model pair, the probe modalities detect **no statistically significant memorization signal** beyond what generic Turkish capability would produce. Formally: for each probe, the perturbation/recall statistic shows **no significant difference** from the contamination-free expectation.

**H1a (verbatim-recognition):** Some frontier models recognize exact TR-MMLU/TUMLU option surfaces at a rate significantly above chance, detectable as *asymmetric* response to the memorized surface vs. a rephrased twin.

**H1b (surface-fragility):** Accuracy collapses when the item surface is perturbed (distractor flip / back-translation) while staying stable for the concept-preserving control — the signature that performance is tied to the memorized string, matching ConStat's "non-generalizing performance" definition [C]. This is the exact pattern the 3-arm harness was built to detect.

**Honest-negative commitment (from the existing result_scale3arm.md):** The pilot-to-n=175 run on DeepSeek v4 Flash / TR-MMLU produced **no significant contamination signal** (English-direct perfectly stable, p=1.00; back-translation directional but p=0.0703, not significant). **This must be treated as a viable, publishable outcome.** A null audit at scale is still a first-of-kind scientific result for Turkish and valuable reported as such. The paper is designed so **H0 is a defensible, citable finding** — not a "failed experiment." The correct reporting discipline is: report every cell honestly, flag directional-but-not-significant results as null, never "unselectively positive" a benchmark.

### 1.4 What a "contaminated benchmark" means here (operational definition)
Adopt ConStat's definition [C]: contamination = performance that is **inflated** relative to behavior on a matched *rephrased/synthetic/control* twin — i.e., **non-generalizing benchmark performance** — rather than "the sample was in training data" (which is unobservable in a black box). This definition is API-observable and is the most defensible framing.

---

## 2. Target benchmarks (all verified live this session)

| Benchmark | arXiv ID | HF repo (verified) | Format | Public since | Role in audit |
|-----------|----------|--------------------|--------|--------------|---------------|
| **TR-MMLU** | 2501.00593 | `alibayram/turkish_mmlu` (gated, CC BY-NC-ND) | MC-QA, 6200 items / 62 sections, `cevap`=int index into `secenekler` | 2024-12-31 | **Primary** (longest exposure, exists in harness) |
| **TUMLU** | 2502.11020 | `jafarisbarov/TUMLU-mini` → `turkish/` (dev+test parquet) | MC-QA, 8 Turkic langs incl. Turkish, mini is "manually verified" subset | 2025-02-16 | MC-QA extension; Turkish split |
| **Halluverse-M³** | 2602.06920 | `sabdalja/HalluVerse-M3` → `FINAL_DATASET/QA/QA_tr.xlsx` | QA + dialogue summarization, 4 langs incl. Turkish | 2026-02-06 | **Novel probe ground**: hallucination edits give a natural "flipped-option" control per item |
| **RAGTurk** | 2602.03652 | `metunlp/ragturk` → `formal_5k`, `informal_6k` | QA + passage chunks; 7 pipeline stages | 2026-02-03 | **Overlap-stat interest** (drawn from Wikipedia/CulturaX — computes corpus-overlap numerator) |
| **TAS** (There Are No Silly Questions) | 2603.09996 | **NOT a public dataset repo** — paper-only | 10 edge-case scenarios, offline LLMs 270M–32B | 2026-02-15 | Qualitatively illustrative only — see caveat below |

**Verification note:** all five arXiv IDs resolve and return the cited metadata; the four HF repos exist with the listed file structures (parquet/excel). TAS is the exception — it is **described in a paper, not released as data**, targets **offline small models (270M–32B)**, and is only 10 scenarios, so it is **not suitable for the statistical matched-subset protocol.** Honest role: TAS enters as a *qualitative verbatim-recall probe* (if we can obtain the scenario text via the paper/appendix/contacting authors) and as a **related-work anchor of the Turkish evaluation wave**, not a scored arm. We will state this explicitly rather than pad the list. Mark `[CITATION NEEDED]` anywhere TAS scenario text is quoted until a primary artifact is secured.

---

## 3. Probe design (three independent modalities — **mandatory**, per [10])

### Why multi-probe is non-negotiable
[10] shows single perturbation probes **degrade at scale**: "Traditional encoder-style probes using perturbations such as synonym fuzzing or dead-code insertion struggle to expose memorization in scaled models." A single probe that returns "no signal" is therefore **not evidence of no contamination** — it may be a broken probe. The audit therefore declares contamination only when **≥2 of 3 independent modalities agree**, and reports each modality separately with its own CI.

### Probe M1 — Verbatim-recall / recognition (surface-exactness)
- For a sample of items, present the **question + options stripped of the answer position ordering** and ask for completion / "do you recognize this?" free-text continuation, measuring **n-gram / token overlap** between the model's output and the ground-truth option.
- Strengthened by the OWL precedent [11]: models recall memorized surface text at measurable rates (GPT-4o identified masked entities 6%/titles 69% on never-trained translations).
- **Metric:** verbatim 8-gram overlap rate (contamination if well above near-zero for novel text) + proportion of items where the model emits the memorized correct option verbatim.

### Probe M2 — Option-perturbation / flipped-distractor & surface-fragility
Two sub-arms, reusing the proven harness (`build_perturbed`, 3-arm protocol):
- **M2a (distractor flip, from existing `run_choice_substitution`):** replace one distractor with another question's correct answer; observe whether the model *follows to the planted "correct"* (concept-independent memorization) or stays correct (concept-generalizing).
- **M2b (surface / cross-lingual fragility, the proven 3-arm run):** Arm A = original Turkish · Arm B = back-translated TR→EN→TR · Arm C = English-direct. The contamination signature is **B collapses while C stays stable ⇒ model anchored to memorized Turkish surface** rather than the underlying concept. This is the ConStat "non-generalizing performance" operationalization [C] and reuses `crosslingual_probe.py`/`scale3arm.py` **verbatim.**
- **Metric:** per-arm accuracy + **paired McNemar exact** versus baseline; the discordant-flip counts are the headline statistic.

### Probe M3 — Overlap statistics against public corpora
- Compute **n-gram overlap** between each benchmark's item text and the web-scale corpora the models plausibly saw (CulturaX-Turkish, Wikipedia tr), mirroring the contamination-report methodology (found 1–45% overlap across 6 benchmarks, ref [26] in §9).
- **Metric:** contiguous 13-gram exact-match rate per benchmark; report as a *descriptive* baseline that contextualizes M1/M2 (does NOT by itself prove training inclusion — black-box limitation, stated).

### Probe triple-derivation table
| Finding requires... | M1 verbatim | M2a flip | M2b fragility | M3 overlap |
|---|---|---|---|---|
| Benchmark flagged contaminated | elevated | + follow-planted | + B-drop | + high overlap |
| Capability (not contamination) | near-zero | correct-resilient | stable all arms | — |
| **Consensus rule:** flag a benchmark only when ≥2 modalities agree AND at least one dynamic arm (M2a/M2b) is significant.

---

## 4. Sampling & matched-subset statistics (reuses the proven harness)

**Proven components reused verbatim** (all exist and ran to n=175+ this project):
- `crosslingual_probe.py` — seed-42 stratified sample, per-item full-match pairing for McNemar
- `analyze_scale3arm.py` — **Wilson 95% CI** (exact interval, `wilson()`), **McNemar exact two-sided** via `stats.binomtest`, honest headline ("no significant contamination signal detected").
- `trmlu_audit/core.py` — client, `parse_letter` (handles reasoning-model verbosity), `build_perturbed`, `roundtrip`.

**Protocol (honesty rules):**
1. **Fixed seed** (42) per benchmark; sample n=200 matched rows per benchmark; report the *fully-matched* denominator (rows where ≥3 arms all parsed) — the n=200→175 drop is already documented and reported in Limitations, not hidden.
2. **Wilson CI** for every accuracy cell; **McNemar exact two-sided** for every paired arm-vs-baseline contrast. α=0.05, **two-sided**.
3. **Multiple-comparison control:** Bonferroni per benchmark×probe family; report both raw and corrected p.
4. **Pre-registered decision rule:** "contaminated" requires ≥2 modalities indicate AND ≥1 dynamic arm p<0.05 after correction.
5. **Reporting:** every cell shown; directional-but-insignificant results labeled **null/suggestive**, never "finding." p>0.05 with overlapping CI = "no significant contamination detected," per result_scale3arm.md wording.
6. **Cost/budget:** API inference only; estimate ≤ ~6,000 answer calls across all cells for the headline study; cheap pilot (n≈50, 1 arm) first, log actual spend (consistent with the scale3arm pilot→scale pattern already proven).

---

## 5. API models to test (3 independent families + reference — LIVE-VERIFIED 2026-08-16)

**Transport change (critical):** OpenRouter is BLOCKED for this account — live-verified 2026-08-16: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `deepseek/deepseek-v4-flash*` all return HTTP 404 "No allowed providers are available for the selected model" (account `allowed-providers` setting). All inference moves to **opencode-go** (`https://opencode.ai/zen/go/v1`, key `OPENCODE_GO_API_KEY` in `C:/Users/oguzc/AppData/Local/hermes/.env`). Pitfall: raw urllib/httpx calls get Cloudflare HTTP 403 unless a browser-like User-Agent is sent; the OpenAI-SDK path (as Hermes uses) works.

| Family (vendor) | Endpoint (verified live 2026-08-16) | Role |
|-----------------|-------------------------------------|------|
| 1. MiniMax | `mimo-v2.5` (basic) | User-picked cheap family member; HTTP 200 verified |
| 2. Moonshot | `kimi-k2.6` | Independent vendor; verified |
| 3. Zhipu | `glm-5.2` | Independent vendor; verified |
| (+reference) | `deepseek-v4-flash` | Already-run null result (paper 1); cohort continuity; verified |
| backups | `kimi-k2.6`, `glm-5.2`, `mimo-v2.5-pro`, `minimax-m3` | Verified live; swap in if a primary degrades |
| EXCLUDED | `qwen3.5-plus`/`qwen3.6-plus` (503 endpoint unavailable, 2026-08-16); ALL OpenRouter models (404 allowlist) | — |

Pilot lineup per user (2026-08-16, UPDATED): **gpt-5.6-luna + mimo-v2.5 + deepseek-v4-flash (reference)** — ALL via opencode-go (user: "so much cheaper there"). mimo-v2.5 + deepseek-v4-flash MUST stay on opencode-go. hy3 REMOVED from lineup — live test 2026-08-16: empty response 3 of 4 calls (unreliable). mimo-v2.5 is intermittently flaky (empty responses ~1/3) → harness MUST retry-on-empty (max 2 retries) and log failures. OpenRouter (provider-fixed by user, ~14¢ credit left): ONLY `openai/gpt-oss-20b` ($0.03/$0.13 per 1M; pilot ≈ $0.01, full run ≈ $0.03) — optional; never gpt-oss-120b. gpt-5.6-luna verified live 2026-08-16 (3/3 solid). **Pure black-box:** no weights, no logits, no hidden states — only sampled text, which is what an API consumer can observe and matches the "black-box" contribution framing.

---

## 6. Expected-results table skeleton (honest-negative interpretation)

Audit scorecard, *expected* shape (values are placeholders to be filled by execution):

| Benchmark | Model family | M1 verbatim | M2a flip-follow | M2b B-vs-C Δ | M3 overlap | Consensus call |
|-----------|--------------|-------------|-----------------|--------------|------------|----------------|
| TR-MMLU | Anthropic | n-gram rate ±CI | p (McNemar) | B-drop ±CI, p | 13-gram % | Contaminated / **No signal** |
| TR-MMLU | Google | … | … | … | … | … |
| TR-MMLU | OpenAI | … | … | … | … | … |
| TUMLU-tr | (families) | … | … | … | … | … |
| Halluverse-M³-tr | (families) | … | … | … | … | (flip-ground given by design) |
| RAGTurk | (families) | … | … | … | corpus-overlap high by construction | interpret carefully |

**Honest-negative example row (TR-MMLU × DeepSeek — already true):**
| M1 low | M2a no follow | M2b C stable / B p=0.0703 | overlap — | **call: no significant contamination** |
M1 low |  — | M2b B-drop arises |  — | **call: directional surface-fragility, NOT significant — report as null with suggestive CI** |

**Overclaim guards:** never call a benchmark "contaminated" on M3 overlap alone (black box can't prove training inclusion); never call it "clean" on any single probe; a null must be phrased "no contamination **detected** by these probes at n=..." — scope-limited, not absolute.

---

## 7. Two-week execution timeline (concrete daily steps)

Weeks of 2026-08-17 .. 2026-08-30 (14 workdays).

**Day 1 (Mon):** Freeze scopes. Read TAS paper (2603.09996) for scenario text; attempt author/artifact contact. Write/pre-register H0/H1 + decision rule in `paper1_contamination_audit.md` top. (Endpoints confirmed DONE 2026-08-16: opencode-go verified, OpenRouter blocked — §5.)
**Day 2 (Tue):** Extend harness to load TUMLU-tr (`jafarisbarov/TUMLU-mini` turkish split) and RAGTurk (`metunlp/ragturk`); verify schema normalization to `{soru, secenekler, cevap}` / QA-pair form. Smoke-test n=5 on 2 models.
**Day 3 (Wed):** Implement M1 verbatim-recall probe + n-gram scorer (reuse token-level overlap). Smoke on TR-MMLU n=20, all 3 families.
**Day 4 (Thu):** Implement M2a (already built — wire to new benchmarks) + M2b 3-arm runner across all target datasets. Dry-run n=10 each to catch parser/format issues.
**Day 5 (Fri):** **Pilot** — n=50 matched per benchmark × 3 families × M1+M2. Compute Wilson/McNemar incl. Bonferroni. Log cost. Gate: if pipeline clean + cost acceptable → proceed.
**Weekend (Sat–Sun):** Analyze pilot; pick final n (target n=200 matched) and models; write Analysis/Results skeleton with honest interpretation framing.

**Day 8 (Mon):** Launch full run in background (batched, rate-limited, per-item streaming JSONL like `scale3arm.jsonl`). Notify on completion.
**Day 9 (Tue):** Run **M3 corpus-overlap** (CulturaX-tr / Wikipedia-tr 13-gram hash comparisons). Can run concurrently with Day-8 inference.
**Day 10 (Wed):** Collect + verify full JSONL; compute all cells, CIs, McNemar, Bonferroni, consensus calls.
**Day 11 (Thu):** Draft Results + Tables §6 filled; write honest-negative interpretations for every cell.
**Day 12 (Fri):** Draft Intro/Related work from the verified reference pack (§9); draft Method §3–4 reusing harness detail.
**Day 13 (Sat):** Draft Discussion/Limitations (black-box limits, TAS non-public caveat, single-null reproducibility); self-revision against honesty rules.
**Day 14 (Sun):** Polish to workshop length; run the `grounded-citations` verify gate on the prose + BibTeX; produce submission package (overleaf `/paper/`).

**Definable deliverable by Day 14:** `notes/partner` analysis + `paper/paper1_contamination_audit.tex` (or .md) + `results/audit_v2.jsonl` + a sources block that passes `verify --evidence`.

---

## 8. Paper skeleton (ACL-collect-latex / workshop length, ~4–6 pages + refs)

**Title:** How Contaminated Are Turkish LLM Benchmarks? A Multi-Probe Memorization Audit of the 2025–2026 Turkish Evaluation Wave
**Abstract** (150 words): claim → method (3 probes) → null-or-finding readiness.

1. **Introduction** — benchmark trust crisis [4][9]; Turkish under-audited; first-of-kind claim; single-sentence contribution: *"we present the first multi-probe, API-only contamination audit of TR-MMLU and the 2025–2026 Turkish benchmark wave, with a pre-registered honest-negative protocol."*
2. **Related Work** — by theme:
   - Contamination & benchmark trust: [27][C][26][R]; leaderboard integrity [9]
   - Turkish resources & known quality gaps: [1][2]a TUMLU / [3] audit / [20] RAGTurk / [19] Halluverse / [22] TAS
   - Probe/memorization methodology & its limits: [11] OWL, [10] scale-aware degradation, [7] rephrased-sample bypass, [12] GAOKAO-Eval
3. **Method** — §3 probes (M1/M2a/M2b/M3) + §4 stats (Wilson + McNemar exact + Bonferroni + consensus rule), reusing the proven harness; black-box scope.
4. **Datasets** — §2 table + verification provenance; TAS exclusion rationale (non-public, offline-target).
5. **Results** — §6 filled cells, per-modality CI, consensus calls, honest negatives featured.
6. **Discussion** — what a null means (scope-limited); surface-fragility direction; recommendations for future Turkish benchmarks (release timestamps, fresh-split, contamination-repeatability per [4] validity windows and WC2026-style contamination-free-by-construction designs [13]).
7. **Limitations** — black-box (no training-data proof); TAS unreleased; per-model n; reasoning-verbosity parse drop (documented n=175/200).
8. **Conclusion** — contribution statement.
**Ethics/Reproducibility** — seeds, full data release plan, HF_TOKEN gating noted, cost log.
**References** — §9 pack, all verified.

**Venue suggestion:** EMNLP 2026 / COLING 2026 workshops (evaluation & trust, low-resource NLP), or ACL Rolling Review co-LREC. NeurIPS 2026 Evaluations & Datasets [6] also strong given the trust theme.

---

## 9. Reference pack (10–15, all verified live with verbatim quotes + arXiv IDs)

### A. Contamination & benchmark trust
- **[C] ConStat — arXiv 2405.16281** (contamination = "artificially inflated and non-generalizing benchmark performance instead of the inclusion of benchmark samples in the training data"). *Method spine of M2.*
- **[9] Who Verifies the Benchmark? — 2608.07762** → "Academic reassessments and independent leaderboards have found undisclosed changes to proprietary models, contaminated training data, and selective reporting."
- **[4] Evaluation Scores Are Perishable Knowledge Claims — 2607.26191** → "validity windows (benchmark results expire as contamination accumulates and distributions shift)."
- **[27] GAOKAO-Eval — 2412.10056** → "there is growing concern that LLMs may ``game'' these benchmarks due to data leakage, achieving high scores while struggling with tasks simple for humans."
- **[26] Open Source Data Contamination Report — 2310.17589** → "varying contamination levels ranging from 1% to 45% across benchmarks, with the contamination degree increasing rapidly over time." (M3 overlap precedent.)
- **[13] WC2026-Agents (contamination-free by construction) — 2607.17765** → "Because every match kicked off after the models' training cutoffs, the benchmark is contamination-free by construction." (positive-control framing in Discussion.)

### B. Probe methodology & memorization
- **[10] Memorization Diagnostics Should Be Scale-Aware — 2608.12771** → "Traditional encoder-style probes using perturbations such as synonym fuzzing or dead-code insertion struggle to expose memorization in scaled models." (*multi-probe justification* — central.)
- **[11] OWL: Cross-Lingual Recall of Memorized Texts — 2505.22945** → "GPT-4o, for example, identifies authors and titles 69% of the time and masked entities 6% of the time in newly translated excerpts." (M1 precedent.)
- **[7] Rephrased-Sample Contamination — 2311.04850** → "simple variations of test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures." (motivates perturbation probe ≥ string-match.)
- **[8] MathShikkha (contamination-audited BanglaMATH) — 2608.08503** → "On the larger, contamination-audited BanglaMATH benchmark, this pattern reverses." (the accepted low-resource audit precedent our paper mirrors for Turkish.)

### C. Turkish resources & quality
- **[1] TR-MMLU — 2501.00593** → "TR-MMLU is constructed from a carefully curated dataset comprising 6200 multiple-choice questions across 62 sections, selected from a pool of 280000 questions." (primary target; exposure since 2024-12-31.)
- **[2] TUMLU — 2502.11020** → "these are conventionally built using machine translation from high-resource languages, which may introduce errors." (native-vs-translated tension.)
- **[3] Benchmark Quality Audit (Turkish) — 2504.09714** → "Our results reveal that 70% of the benchmark datasets fail to meet our heuristic quality standards."
- **[19] RAGTurk — 2602.03652** → "design guidance remains English-centric, limiting insights for morphologically rich languages like Turkish." (wave inclusion + overlap-numerator source: Wikipedia/CulturaX.)
- **[21] Halluverse-M³ — 2602.06920** → "Halluverse-M^3 covers four languages, English, Arabic, Hindi, and Turkish." (flipped-option ground by construction.)
- **[22] There Are No Silly Questions / TAS — 2603.09996** → "a Turkish Anomaly Suite (TAS) consisting of 10 original edge-case scenarios was developed." (wave anchor; not scored.)

*(15 refs; all quotes appear verbatim in the fetched arXiv abstracts this session. BibTeX + eprint generation script is in the `arxiv` skill — run at submission to produce `.bib`.)*

---

## Appendix — verifiable provenance this session
- arXiv API `id_list` fetch returned 14/14 papers (1412.10056→2607.17765 range), full abstracts captured.
- HF API: `alibayram/turkish_mmlu` (dl=138, likes=68), `jafarisbarov/TUMLU-mini` (`turkish/dev+test.parquet`), `metunlp/ragturk` (`formal_5k+informal_6k`), `sabdalja/HalluVerse-M3` (`QA_tr.xlsx` at `FINAL_DATASET/QA/`). ConStat 2405.16281 + rephrase 2311.04850 + contamination report 2310.17589 all resolved.
- TAS: no public data repo detected → handled with `[CITATION NEEDED]` discipline, not padded.
- Existing null result (result_scale3arm.md) held up as the honest-negative exemplar the whole paper design protects.