# tr-contamination-audit

Multi-benchmark contamination audit of Turkish LLM benchmarks (TR-MMLU, TUMLU-tr, RAGTurk, Halluverse-M3) using three independent probe modalities (M1 verbatim recall, M2 option-perturbation/surface-fragility, M3 corpus-overlap) against black-box API models (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash via the opencode-go transport).

This is the artifact repository for the multi-benchmark Turkish contamination audit (the sibling of [oguzcura/trmlu-audit](https://github.com/oguzcura/trmlu-audit), which covers the TR-MMLU paper only). It holds the frozen pre-registration (hypotheses, decision rules), the probe harness, the verified dataset snapshots, and full n=200 results.

**Status: FULL RUN COMPLETE (2026-08-17).** 13,193 API calls, 0/12 cells flagged under M1-DISCOUNT. See `paper/paper2_workshop_draft.md` for the workshop paper and `notes/full_results_2026-08-16.md` for the full report.

## Repository structure

| Path | Contents |
|------|----------|
| `pre-registration.md` | Frozen design doc: H0/H1, probe definitions M1/M2/M3, sampling & matched-subset statistics, decision rule (frozen 2026-08-16) |
| `harness/` | Probe harness: dataset builder, M1 probe, cross-lingual 3-arm probe, analyzers |
| `data/` | Verified dataset snapshots (CSV) + `manifest.json`; see `data/README.md` |
| `harness/results/` | Full n=200 audit trail: `full_audit_2026-08-16.jsonl` (13,198 lines), `full_stats_2026-08-16.json`, spend logs, digest |
| `notes/` | Results report, pilot results, pre-registration, verification pass |
| `paper/` | Workshop draft (`paper2_workshop_draft.md`), skeleton, README |

## Key results

| Benchmark | gpt-5.6-luna | mimo-v2.5 | deepseek-v4-flash |
|---|---|---|---|
| TR-MMLU | ✅ No contamination | ✅ No contamination | ✅ No contamination |
| TUMLU-tr | ✅ No contamination | ✅ No contamination | ⚠️ THRESHOLD-DEPENDENT |
| Halluverse-M3-tr | ✅ No contamination | ✅ No contamination | ✅ No contamination |
| RAGTurk formal_5k | ✅ No contamination | ✅ No contamination | ✅ No contamination |

0/12 cells flagged under the pre-registered M1-DISCOUNT rule. One RAW-only flag (TUMLU-tr × deepseek-v4-flash) is THRESHOLD-DEPENDENT (M1-sensitive), never "contaminated."

## Probes (per pre-registration §3)

- **M1 — Verbatim-recall / recognition:** shuffled-option verbatim reproduction; 8-gram overlap signature. (`harness/probe_m1.py`)
- **M2 — Option-perturbation / surface-fragility:** M2a flipped distractor, M2b cross-lingual back-translation arms A/B/C. (`harness/crosslingual_probe.py`)
- **M3 — Overlap statistics:** contiguous 13-gram exact-match against public Turkish corpora (CulturaX-tr, Wikipedia-tr); descriptive baseline only. *(Not executed in full run — see limitations.)*

## Reproduction

Requires Python ≥3.10 (uv recommended) and the model API key:

```bash
uv sync
# The opencode-go key is read from the Hermes .env (OPENCODE_GO_API_KEY) by
# harness/src/trmlu_audit/core.py. It is NEVER committed.

# 1. Regenerate dataset CSVs + manifest from the Hugging Face sources (no API spend)
uv run python harness/datasets_v2.py

# 2. M1 verbatim-recall smoke (matches results/smoke_2026-08-16.jsonl)
uv run python harness/probe_m1.py --model deepseek-v4-flash --limit 20 --phase m1_smoke --log results/smoke_2026-08-16.jsonl

# 3. Cross-lingual 3-arm run (A baseline / B back-translation / C English-direct)
uv run python harness/crosslingual_probe.py --limit 200 --arms A B C --out results/scale3arm.jsonl

# 4. Full n=200 audit (4 benchmarks × 3 models × probes M1/M2a/M2b)
uv run python harness/full_audit.py --limit 200 --max-cost 2.00 --log results/full_audit_2026-08-16.jsonl

# 5. Analyzers
uv run python harness/analyze_scale3arm.py
uv run python harness/analyze_pilot.py
uv run python harness/analyze_full_audit.py
```

## Spend

| Run | gpt-5.6-luna | deepseek-v4-flash | mimo-v2.5 | Total |
|---|---|---|---|---|
| Smoke | $0.001 | $0.002 | $0.000 | $0.003 |
| Pilot (n=50) | $0.085 | $0.091 | $0.000 | $0.176 |
| Full (n=200) | $0.348 | $0.358 | $0.000 | $0.707 |
| **TOTAL** | **$0.434** | **$0.452** | **$0.000** | **$0.886** |

mimo-v2.5 is uncosted (no published rate on opencode-go). 26 spend checkpoints logged; no cost cap breach.

## Known honest limitations

- **Black-box API:** training-data inclusion is unobservable; probes infer memorization from sampled text behavior. Weight-based methods would require model access.
- **M1 confound:** verbatim-recall probe is confounded by instruction-following; handled by pre-registered M1-DISCOUNT reading (RAW shown for transparency).
- **QA matcher:** RAGTurk matcher is precision-first (1.000 precision, 0.371 recall on 106 labeled pairs); every RAGTurk accuracy is a conservative lower bound.
- **Sample size:** n=200 per benchmark × model is a sample, not the full benchmark.
- **Translation noise:** RAGTurk Arm C golds are machine-translated; translation error propagates into C scoring.
- **Model scope:** only 3 models tested; other models deployed to Turkish users may differ.
- **Benchmark scope:** only 4 benchmarks audited; other Turkish datasets may have different profiles.
- **No M3:** corpus-overlap probe was pre-registered but not executed (cost/access constraints).
- **Run interruptions:** 3 crashes during full run; all resume logic logged, no data lost.

## License

MIT — see [LICENSE](LICENSE).
