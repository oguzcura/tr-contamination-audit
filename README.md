# tr-contamination-audit

Multi-benchmark contamination audit of Turkish LLM benchmarks (TR-MMLU, TUMLU-tr, RAGTurk, Halluverse-M3) using three independent probe modalities (M1 verbatim recall, M2 option-perturbation/surface-fragility, M3 corpus-overlap) against black-box API models (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash via the opencode-go transport).

This is the artifact repository for the multi-benchmark Turkish contamination audit (the sibling of [oguzcura/trmlu-audit](https://github.com/oguzcura/trmlu-audit), which covers the TR-MMLU paper only). It holds the frozen pre-registration (hypotheses, decision rules), the probe harness, the verified dataset snapshots, and real smoke results.

**Status: pilot phase, full results pending.** The files in this repository are real and reproducible; the full M1/M2/M3 run across all benchmarks × models has not completed yet and no benchmark-containment claims are made from the pilot data. See [pre-registration.md](pre-registration.md) for the frozen H0/H1 and the consensus decision rule that will be applied to the full run.

## Repository structure

| Path | Contents |
|------|----------|
| `pre-registration.md` | Frozen design doc: H0/H1, probe definitions M1/M2/M3, sampling & matched-subset statistics, decision rule, two-week plan (frozen 2026-08-16) |
| `harness/` | Probe harness: dataset builder, M1 probe, cross-lingual 3-arm probe, analyzers |
| `harness/src/trmlu_audit/core.py` | opencode-go transport (`chat_record`: retry-on-empty, content+reasoning parsing, usage capture) — reused from trmlu-audit |
| `data/` | Verified dataset snapshots (CSV) + `manifest.json`; see `data/README.md` |
| `results/` | Real run output; currently `smoke_2026-08-16.jsonl` (70 M1/BB smoke records, deepseek-v4-flash) |
| `paper/` | LaTeX source + PDF for the paper (lands when the full run completes) |

## Probes (per pre-registration §3)

- **M1 — Verbatim-recall / recognition:** shuffled-option verbatim reproduction; 8-gram overlap signature. (`harness/probe_m1.py`)
- **M2 — Option-perturbation / surface-fragility:** M2a flipped distractor, M2b cross-lingual back-translation arms A/B/C. (`harness/crosslingual_probe.py`)
- **M3 — Overlap statistics:** contiguous 13-gram exact-match against public Turkish corpora (CulturaX-tr, Wikipedia-tr); descriptive baseline only.

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

# 3. Black-box letter baseline smoke (transport check)
uv run python harness/probe_m1.py --model deepseek-v4-flash --limit 20 --bb --log results/smoke_2026-08-16.jsonl

# 4. Cross-lingual 3-arm run (A baseline / B back-translation / C English-direct)
uv run python harness/crosslingual_probe.py --limit 200 --arms A B C --out results/scale3arm.jsonl

# 5. Analyzers: accuracy, Wilson CI, McNemar exact
uv run python harness/analyze_scale3arm.py
uv run python harness/analyze_pilot.py
```

## Known honest limitations

- `harness/crosslingual_probe.py` still uses the legacy OpenRouter transport (`OPENROUTER_API_KEY`); the account has verified OpenRouter restrictions (404 for all providers), so the port to `trmlu_audit.core` (opencode-go) is pending. The analyzer and probe files are real and were exercised during development, but only the M1/BB path has produced committed results so far.
- `data/ragturk_informal6k.csv` is header-only: the Hugging Face informal_6k split contains no data files (verified live 2026-08-16, see `data/README.md` and `manifest.json`).
- Black-box probes cannot prove training inclusion; M3 overlap is descriptive only. No benchmark is called "contaminated" or "clean" on any single probe (overclaim guards in pre-registration §6).

## License

MIT — see [LICENSE](LICENSE).
