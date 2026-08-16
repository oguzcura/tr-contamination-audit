# Paper 1 — Day 1: OpenRouter Endpoint Verification

**Date:** 2026-08-16 · **Method:** `GET /api/v1/models` via OpenRouter (key from `C:\Users\oguzc\AppData\Local\hermes\.env`, `OPENROUTER_API_KEY`, never echoed).
**Result:** all four requested endpoints **resolve (HTTP 200)** on OpenRouter. Exact IDs recorded below for the harness — these are the strings that must be passed to `client.chat.completions.create(model=...)`.

## Verified endpoints

| Requested family | Exact live `model` id | OpenRouter ctx tokens | Pricing (USD / token, from /models) |
|---|---|---|---|
| Anthropic | `anthropic/claude-opus-5-fast` | 1,000,000 | prompt 1.0e-5 ($10/M) · completion 5.0e-5 ($50/M) |
| Google | `google/gemini-3.7-flash` | 1,048,576 | prompt 3.75e-7 ($0.375/M) · completion 1.875e-6 ($1.875/M) |
| OpenAI | `openai/gpt-5.6-luna` | 1,050,000 | prompt 1.0e-7 ($0.10/M) · completion 6.0e-7 ($0.60/M) |
| DeepSeek (reference) | `deepseek/deepseek-v4-flash` | 1,048,576 | prompt 6.426e-8 ($0.06426/M) · completion 1.2852e-7 ($0.12852/M) |

Notes:
- DeepSeek has an additional dated alias live: `deepseek/deepseek-v4-flash-0731` (already used by the existing harness, `crosslingual_probe.py` / `scale3arm.py`). The requested bare id `deepseek/deepseek-v4-flash` **also resolves exactly** — that is the id used by `probe_m1.py`.
- Pricing table is pulled live from `/models`; it does **not** list a separate `cache_read` line under `pricing`, so cache-input rates are taken from the OpenRouter model listing convention and logged per the Day-3 spec below.
- All four are in the 400+ model catalogue (total returned: 413 models).

## Provider-routing caveat (observed during Day 3 smoke)
This OpenRouter account's **allowed-providers setting permits only `decart`**.
At execution time the bare `deepseek/deepseek-v4-flash` id's available provider
set did NOT include an allowed provider, so an inference call returned
`404 no allowed providers` (the id still *resolves* on `/models`; routing is a
separate runtime constraint). The dated alias `deepseek/deepseek-v4-flash-0731`
IS served by an allowed provider and produced a clean M1 response on the same
call. Therefore the harness reference id used for actual inference is
`deepseek/deepseek-v4-flash-0731` (identical pricing ladder; this is also the id
the pre-existing `crosslingual_probe.py` / `scale3arm.py` already use).
**Verify provider routing per family before the Day-4/5 pilot** — the same
decart-only constraint may hit Anthropic/Google/OpenAI endpoints too.

## Cost-accounting constants used for M1 smoke (Day 3, per task spec)

These are the fixed ladder used by `probe_m1.py` cost logging for `deepseek/deepseek-v4-flash`:

- input: **$0.07 / M token**
- cache-read: **$0.0014 / M token**
- output: **$0.14 / M token**

Cost per call = (prompt_tokens/1e6 × 0.07) + (cached_tokens/1e6 × 0.0014) + (completion_tokens/1e6 × 0.14).
Token counts are read from the OpenRouter `usage` object returned on each chat completion (prompt_completion split; the `cache_read_input_tokens` field when present is charged at the cache rate, otherwise 0).

## Correctness check against live `/models` figures
The task-specified ladder ($0.07/$0.0014/$0.14 per M) is close to the resolved DeepSeek listing (prompt $0.06426/M, completion $0.12852/M). As instructed, the **fixed spec ladder** governs the smoke cost log to keep numbers comparable and reproducible; the resolved `prompt`/`completion` values above are recorded as ground truth.