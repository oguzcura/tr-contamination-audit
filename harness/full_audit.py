"""FULL n=200 runner for the Paper 1 contamination audit (Day 8, 2026-08-16).

Same transport/design as the Day-5 pilot runner (opencode-go only via
trmlu_audit.core; OpenRouter untouched) with the four USER-MANDATED fixes:

  FIX 1 (pre-registration): the M1-discount sensitivity adjudication is
         pre-registered in notes/pre_reg_full_2026-08-16.md AND in the report
         notes/full_results_2026-08-16.md BEFORE this run starts. The analyzer
         reports BOTH raw-threshold (§0) and M1-discount consensus readings.
  FIX 2 (retry-harder): gpt-5.6-luna on RAGTurk long/recurring items gets up to
         4 retries with exponential backoff [2,4,8,16]s. Triggers, logged in
         field retry_hard: "long" (surface > 2000 chars; none exist in
         formal_5k - max 1583 - kept for spec fidelity) or "recurring-item"
         (the item already failed once for luna on RAGTurk in the current log
         or in the resumed log). Every failure is logged with its error.
  FIX 3 (answer-form-robust RAGTurk scoring): qa_matcher.py cascade
         (exact / letter / normalized-paraphrase, number-guarded, distinctive-
         gram + content-token guarded; measured precision 1.000 / recall 0.371
         on 106 hand-labeled pilot pairs). Arm-C gold answers are machine
         translated TR->EN once per item (probe M2b_gold_en, deepseek, logged)
         so the English arm is scored against an English gold. Legacy lenient
         containment and a lenient matcher sensitivity are logged per call.
  FIX 4 (cost): --max-cost 2.00 hard-abort (dry-run estimate + live abort, as
         in the pilot) PLUS a spend checkpoint line appended to
         results/spend_full_2026-08-16.jsonl every 500 logged calls.

Hardening inherited from the pilot: retry-on-empty (core), 5xx/429 backoff
(max 2, logged), append-mode JSONL + RESUME-SKIP by item_key, workers=4,
global ~2 req/s token bucket, translations computed once per (benchmark, item)
and cached + re-seeded from the log on resume.

Run (from harness/):
  uv run python full_audit.py --dry-run
  uv run python full_audit.py --limit 200 --workers 4
             --log results/full_audit_2026-08-16.jsonl
  (same command re-run resumes where it stopped)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "trmlu-audit", "src"))
from trmlu_audit import core  # noqa: E402
from probe_m1 import overlap8, is_verbatim, cost_estimate, LADDER  # noqa: E402
from qa_matcher import matcher as qa_matcher  # noqa: E402

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_LOG = os.path.join(HERE, "results", "full_audit_2026-08-16.jsonl")
DEFAULT_SPEND = os.path.join(HERE, "results", "spend_full_2026-08-16.jsonl")

TRANS_MODEL = core.TRANS_MODEL          # deepseek-v4-flash
ANS_MODELS = ["gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"]
BENCHMARKS = ["tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"]
RETRY_HARD_MODEL = "gpt-5.6-luna"       # fix 2 scope
RETRY_HARD_BENCH = "ragturk_formal5k"
RETRY_HARD_MAX = 4                      # 1 initial + 4 retries = 5 attempts
RETRY_HARD_BACKOFF = [2.0, 4.0, 8.0, 16.0]
LONG_SURFACE_CHARS = 2000               # spec-fidelity threshold (fix 2)

# ---------------------------------------------------------------------------
# Module-level run state (shared by workers + lazy translation caches)
# ---------------------------------------------------------------------------
BUCKET: "TokenBucket" = None
MAX_COST = [2.00]
SPENT = [0.0]
ABORT = threading.Event()
LOG_PATH = [DEFAULT_LOG]
SPEND_PATH = [DEFAULT_SPEND]
LOG_LOCK = threading.Lock()
COST_LOCK = threading.Lock()
CALLS_SINCE_CHECKPOINT = [0]

_FAILED_LUNA_ITEMS: set = set()         # (benchmark, item_idx) that ever failed
_FAILED_ITEMS_LOCK = threading.Lock()


def _checkpoint():
    """Spend log line every 500 logged calls (fix 4).

    LOCK CONTRACT: never acquires COST_LOCK itself (non-reentrant). Callers:
    log_record (already holds COST_LOCK) or main() after the pool exits
    (exclusive). Writes the file under LOG_LOCK.
    """
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "calls_logged": CALLS_SINCE_CHECKPOINT[0],
           "accumulated_cost_est_usd": round(SPENT[0], 6),
           "cost_cap_usd": MAX_COST[0],
           "abort": ABORT.is_set()}
    with LOG_LOCK:
        with open(SPEND_PATH[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"[spend-chk] calls={rec['calls_logged']} cost≈${rec['accumulated_cost_est_usd']:.4f}")


def log_record(rec: dict):
    """Thread-safe append of one call record + live cost accounting (fix 4)."""
    with LOG_LOCK:
        with open(LOG_PATH[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    with COST_LOCK:
        CALLS_SINCE_CHECKPOINT[0] += 1
        if CALLS_SINCE_CHECKPOINT[0] % 500 == 0:
            _checkpoint()
        if rec.get("cost_est_usd") is not None:
            SPENT[0] += rec["cost_est_usd"]
            if SPENT[0] > MAX_COST[0] and not ABORT.is_set():
                ABORT.set()
                print(f"\nHARD ABORT (live): accumulated estimated cost "
                      f"${SPENT[0]:.4f} exceeded ${MAX_COST[0]:.2f}. "
                      f"Remaining calls not executed. Log is intact for resume.")
    if not rec["ok"] and rec.get("model") == RETRY_HARD_MODEL:
        with _FAILED_ITEMS_LOCK:
            _FAILED_LUNA_ITEMS.add((rec.get("benchmark"), rec.get("item_idx")))


def log_event(event: str, extra: Optional[dict] = None):
    """Control-plane marker line in the JSONL (run_start / resume / abort)."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
    if extra:
        rec.update(extra)
    with LOG_LOCK:
        with open(LOG_PATH[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()


# ---------------------------------------------------------------------------
# Data loading (normalized CSVs built by datasets_v2.py, verified Day 2-3)
# ---------------------------------------------------------------------------
def _load_csv(name: str) -> List[Dict]:
    import csv as _csv
    path = os.path.join(DATA_DIR, name)
    out = []
    with open(path, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            out.append({k: (v or "") for k, v in r.items()})
    return out


def _parse_list(s: str) -> List[str]:
    if isinstance(s, list):
        return list(s)
    s = s.strip()
    try:
        val = json.loads(s)
        return list(val) if isinstance(val, list) else [str(val)]
    except Exception:
        import ast
        try:
            val = ast.literal_eval(s)
            return list(val) if isinstance(val, (list, tuple)) else [str(val)]
        except Exception:
            return [c.strip().strip("'\"") for c in s.strip("[]()").split(",") if c.strip()]


def load_sample(benchmark: str, limit: int, seed: int = SEED) -> List[Dict]:
    """Seed-42 sample shared by ALL models and probes (matched design)."""
    raw = _load_csv(f"{benchmark}.csv")
    rng = random.Random(seed)
    rows = rng.sample(raw, min(limit, len(raw)))
    out = []
    for i, r in enumerate(rows):
        base = {"idx": i, "source": benchmark, "soru": r["soru"]}
        if benchmark in ("tr_mmlu", "tumlu_tr"):      # MC form
            opts = _parse_list(r["secenekler"])
            try:
                idx = int(r["cevap_idx"]) if r.get("cevap_idx") else None
            except ValueError:
                idx = None
            if idx is None or not (0 <= idx < len(opts)):
                letter = r.get("cevap", "").strip().upper()
                idx = ord(letter) - ord("A") if letter in "ABCDE" else None
            base.update({"secenekler": opts, "cevap_idx": idx,
                         "cevap": r.get("cevap", "").strip().upper(),
                         "bolum": r.get("bolum", r.get("subject", ""))})
        else:                                          # QA-pair form
            base.update({"cevap": r["cevap"],
                         "edited_cevap": r.get("edited_cevap", ""),
                         "qid": r.get("qid", ""), "category": r.get("category", "")})
        out.append(base)
    return out


# ---------------------------------------------------------------------------
# Prompt builders (identical to the pilot)
# ---------------------------------------------------------------------------
def mc_text(row: Dict, options: Optional[List[str]] = None) -> str:
    opts = options if options is not None else row["secenekler"]
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(opts)))


def build_m1_mc_prompt(row: Dict, idx: int):
    opts = list(row["secenekler"])
    truth_idx = row["cevap_idx"]
    truth_text = opts[truth_idx] if truth_idx is not None and truth_idx < len(opts) else ""
    rng = random.Random(SEED + idx)
    order = list(range(len(opts)))
    rng.shuffle(order)
    shuffled = [opts[i] for i in order]
    lines = [f"Soru: {row['soru']}", "", "Secenekler:"]
    lines += [f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled)]
    prompt = "\n".join(lines) + (
        "\n\nCevap (ONLY the letter of the correct choice, e.g. A) followed by "
        "the FULL EXACT text of that choice, reproduced verbatim as it appears above:")
    return prompt, truth_text


def build_m1_qa_prompt(row: Dict) -> str:
    return ("Soru: " + row["soru"] +
            "\n\nBu sorunun doğru cevabını, kaynak metindeki haliyle AYNEN (birebir) "
            "yazın. Sadece cevap metnini verin:")


def build_m2a_mc_prompt(rows: List[Dict], i: int):
    row = rows[i]
    correct = row["secenekler"][row["cevap_idx"]]
    other = rows[(i + 1) % len(rows)]
    other_ans = other["secenekler"][other["cevap_idx"]]
    pert = list(row["secenekler"])
    planted_letter = None
    for k, c in enumerate(pert):
        if c != correct:
            pert[k] = other_ans
            planted_letter = chr(65 + k)
            break
    if planted_letter is None or other_ans == correct:
        return None, None, None, None
    return mc_text(row, pert), row["cevap"], planted_letter, other_ans


def build_m2a_hv_prompt(row: Dict, idx: int):
    truth, edited = row["cevap"].strip(), row["edited_cevap"].strip()
    if not truth or not edited or truth == edited:
        return None, None
    rng = random.Random(SEED + idx)
    if rng.random() < 0.5:
        a, b, truth_letter, edited_letter = truth, edited, "A", "B"
    else:
        a, b, truth_letter, edited_letter = edited, truth, "B", "A"
    prompt = ("Soru: " + row["soru"] +
              f"\n\nHangisi doğru cevaptır?\nA) {a}\nB) {b}\n\nCevap (tek harf A/B):")
    return prompt, truth_letter, edited_letter


# ---------------------------------------------------------------------------
# QA answer scoring (fix 3): robust matcher + legacy lenient for sensitivity
# ---------------------------------------------------------------------------
def _norm_legacy(s: str) -> str:
    s = re.sub(r"\s+", " ", s.lower()).strip(" .,;:!?\"'")
    return s


def qa_legacy_containment(gold: str, pred: str) -> bool:
    g, p = _norm_legacy(gold), _norm_legacy(pred)
    if not g or not p:
        return False
    return g == p or g in p or p in g


def qa_score_robust(gold: str, pred: str, question: str):
    """(strict_match, level, sim, lenient_match, legacy_containment)."""
    m, lev, sim = qa_matcher(gold, pred, question, strict=True)
    lm, _, _ = qa_matcher(gold, pred, question, strict=False)
    return (bool(m), lev, round(sim, 3), bool(lm), qa_legacy_containment(gold, pred))


# ---------------------------------------------------------------------------
# Token bucket rate limiter (thread-safe, global ~2 req/s)
# ---------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, rate: float, capacity: float = 4.0):
        self.rate = rate
        self.cap = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.cap,
                                  self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(max(wait, 0.01))


# ---------------------------------------------------------------------------
# Retry layers: (a) core retry-on-empty (max 2); (b) 5xx/429 backoff (max 2);
# (c) FIX 2 retry-hard for gpt-5.6-luna x RAGTurk long/recurring items (max 4)
# ---------------------------------------------------------------------------
RETRY_RE = re.compile(r"(429|5\d\d|rate.?limit|overload|too many|timeout|timed out|"
                      r"ConnectionError|connection|ECONN|503|502|500)", re.I)


def _retry_hard_reason(benchmark: str, model: str, surface_len: int, item: int) -> Optional[str]:
    if model != RETRY_HARD_MODEL or benchmark != RETRY_HARD_BENCH:
        return None
    if surface_len > LONG_SURFACE_CHARS:
        return "long"
    with _FAILED_ITEMS_LOCK:
        if (benchmark, item) in _FAILED_LUNA_ITEMS:
            return "recurring-item"
    return None


def call_with_retries(client, model: str, system: str, user: str,
                      max_tokens: int = 1000, max_http_retries: int = 2,
                      benchmark: str = "", item: int = -1,
                      retry_hard_max: int = RETRY_HARD_MAX):
    """core.chat_record + http-backoff retries (+ fix-2 retry-hard path)."""
    reason = _retry_hard_reason(benchmark, model, len(user), item)
    retry_log: List[Dict] = []
    if reason:
        # retry-hard: up to RETRY_HARD_MAX extra attempts with backoff on top of
        # core's empty-retries (chat_record already does its own 2)
        rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
        attempt = 0
        while (not rec["ok"]) and attempt < retry_hard_max:
            attempt += 1
            backoff = RETRY_HARD_BACKOFF[min(attempt, len(RETRY_HARD_BACKOFF)) - 1]
            retry_log.append({"attempt": attempt, "error": rec["error"],
                              "backoff_s": backoff, "retry_hard_reason": reason})
            print(f"[retry-hard] {model} {benchmark} item={item} reason={reason} "
                  f"attempt={attempt} backoff={backoff}s err={(rec['error'] or '')[:100]}")
            time.sleep(backoff)
            rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
        rec["retry_log"] = retry_log
        rec["retries"] = rec["retries"] + len(retry_log)
        rec["retry_hard"] = reason
        return rec
    rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
    attempt = 0
    while (not rec["ok"]) and attempt < max_http_retries and RETRY_RE.search(rec["error"] or ""):
        attempt += 1
        backoff = 2.0 * (2 ** (attempt - 1))
        retry_log.append({"attempt": attempt, "error": rec["error"], "backoff_s": backoff})
        print(f"[http-retry] {model} attempt={attempt} backoff={backoff}s "
              f"err={rec['error'][:120]}")
        time.sleep(backoff)
        rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
    rec["retry_log"] = retry_log
    rec["retries"] = rec["retries"] + len(retry_log)
    rec["retry_hard"] = None
    return rec


# ---------------------------------------------------------------------------
# Translation caches: TR->EN->TR surfaces computed ONCE per (benchmark, item);
# RAGTurk arm-C gold is translated TR->EN once per item (fix 3).
# ---------------------------------------------------------------------------
_TRANS_LOCK = threading.Lock()
_TRANS_CACHE: Dict[tuple, Dict[str, str]] = {}
_GOLD_EN_LOCK = threading.Lock()
_GOLD_EN_CACHE: Dict[int, str] = {}


def _ensure_translations(client, benchmark: str, row: Dict, item: int):
    """{'backtr_tr','en_direct'} for (benchmark, item); thread-safe + logged."""
    key = (benchmark, item)
    with _TRANS_LOCK:
        if key in _TRANS_CACHE:
            return _TRANS_CACHE[key]
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            src = mc_text(row)
        else:
            src = "Soru: " + row["soru"]
        sys_tr_en = ("You are a faithful professional Turkish->English translator. "
                     "Translate the whole question AND every choice. Keep letter "
                     "labels (A,B,C,..) as-is. Preserve meaning precisely. "
                     "Output ONLY the translation.")
        sys_en_tr = ("You are a faithful professional English->Turkish translator. "
                     "Translate back to Turkish. Keep letter labels (A,B,C,..) as-is. "
                     "Preserve meaning precisely. Output ONLY the translation.")
        BUCKET.acquire()
        rec_en = call_with_retries(client, TRANS_MODEL, sys_tr_en, src, max_tokens=700,
                                   benchmark=benchmark, item=item)
        log_record(_record(benchmark, TRANS_MODEL, "M2b_tr_en", item, rec_en,
                           {"translation": rec_en["content"][:1000], "arm": "tr_en"}))
        BUCKET.acquire()
        rec_tr = call_with_retries(client, TRANS_MODEL, sys_en_tr,
                                   rec_en["content"], max_tokens=700,
                                   benchmark=benchmark, item=item)
        log_record(_record(benchmark, TRANS_MODEL, "M2b_en_tr", item, rec_tr,
                           {"translation": rec_tr["content"][:1000], "arm": "en_tr"}))
        out = {"backtr_tr": rec_tr["content"], "en_direct": rec_en["content"]}
        _TRANS_CACHE[key] = out
        return out


def _ensure_gold_en(client, row: Dict, item: int) -> str:
    """RAGTurk arm-C English gold (fix 3); one logged call per item."""
    with _GOLD_EN_LOCK:
        if item in _GOLD_EN_CACHE:
            return _GOLD_EN_CACHE[item]
        sys_ = ("You are a faithful professional Turkish->English translator. "
                "Translate this answer text to English. Output ONLY the translation.")
        BUCKET.acquire()
        rec = call_with_retries(client, TRANS_MODEL, sys_, row["cevap"],
                                max_tokens=700, benchmark="ragturk_formal5k", item=item)
        log_record(_record("ragturk_formal5k", TRANS_MODEL, "M2b_gold_en", item, rec,
                           {"translation": rec["content"][:1000], "arm": "gold_en"}))
        _GOLD_EN_CACHE[item] = rec["content"] if rec["ok"] else ""
        return _GOLD_EN_CACHE[item]


# ---------------------------------------------------------------------------
# Probe execution -> one dict of outcome fields per call
# ---------------------------------------------------------------------------
def run_probe_call(client, benchmark: str, model: str, rows: List[Dict],
                   probe: str, item: int) -> Optional[dict]:
    row = rows[item]
    if probe == "M1":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            if row["cevap_idx"] is None:
                return None
            prompt, truth_text = build_m1_mc_prompt(row, item)
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Give ONLY the letter and then the verbatim choice text.")
            truth = truth_text
        else:
            prompt, truth = build_m1_qa_prompt(row), row["cevap"]
            system = ("You are answering a Turkish factual QA task. Reproduce the "
                      "exact answer text verbatim, nothing else.")
        rec = call_with_retries(client, model, system, prompt, max_tokens=700,
                                benchmark=benchmark, item=item)
        out = {"overlap8": round(overlap8(truth, rec["content"]), 4),
               "verbatim_hit": is_verbatim(truth, rec["content"]),
               "truth_text": truth[:400]}
        return _record(benchmark, model, probe, item, rec, out)

    if probe == "M2a":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            prompt, truth_letter, planted_letter, planted_text = build_m2a_mc_prompt(rows, item)
            if prompt is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    prompt + "\n\nCevap (tek harf A/B/C/D/E):",
                                    benchmark=benchmark, item=item)
            parsed = core.parse_letter(rec["content"])
            correct = parsed == truth_letter
            flip = bool(parsed == planted_letter) and planted_text != row["secenekler"][row["cevap_idx"]]
            out = {"parsed_letter": parsed, "answer_correct": correct,
                   "flip_follow": flip, "truth_letter": truth_letter,
                   "planted_letter": planted_letter}
        else:  # halluverse: edited-answer flip (2-option forced choice)
            prompt, truth_letter, edited_letter = build_m2a_hv_prompt(row, item)
            if prompt is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B).")
            rec = call_with_retries(client, model, system, prompt, max_tokens=400,
                                    benchmark=benchmark, item=item)
            parsed = core.parse_letter(rec["content"])
            correct = parsed == truth_letter
            flip = bool(parsed == edited_letter)
            out = {"parsed_letter": parsed, "answer_correct": correct,
                   "flip_follow": flip, "truth_letter": truth_letter,
                   "planted_letter": edited_letter,
                   "edited_cevap": row.get("edited_cevap", "")[:200]}
        return _record(benchmark, model, probe, item, rec, out)

    # ---------------- M2b arms (translations via TRANS_MODEL) ----------------
    if probe == "M2b_A":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            if row["cevap_idx"] is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    mc_text(row) + "\n\nCevap (tek harf A/B/C/D/E):",
                                    benchmark=benchmark, item=item)
            parsed = core.parse_letter(rec["content"])
            out = {"parsed_letter": parsed,
                   "answer_correct": parsed == row["cevap"], "arm": "A"}
        else:
            system = ("You are answering a Turkish factual QA question. Give the "
                      "answer in Turkish, concise and exact.")
            rec = call_with_retries(client, model, system,
                                    "Soru: " + row["soru"] + "\n\nCevap:",
                                    benchmark=benchmark, item=item)
            match, lev, sim, lmatch, legacy = qa_score_robust(
                row["cevap"], rec["content"], row["soru"])
            out = {"qa_match": match, "qa_level": lev, "qa_sim": sim,
                   "qa_match_lenient": lmatch, "qa_legacy_containment": legacy,
                   "answer_correct": match, "arm": "A"}
        return _record(benchmark, model, probe, item, rec, out)

    if probe in ("M2b_B", "M2b_C"):
        surf = _ensure_translations(client, benchmark, row, item)
        surface = surf["backtr_tr"] if probe == "M2b_B" else surf["en_direct"]
        if not surface:
            return None                       # translation call failed earlier
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            system = ("You are taking a standardized multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    surface + "\n\nCevap (tek harf A/B/C/D/E):",
                                    benchmark=benchmark, item=item)
            parsed = core.parse_letter(rec["content"])
            out = {"parsed_letter": parsed,
                   "answer_correct": parsed == row["cevap"],
                   "arm": "B" if probe == "M2b_B" else "C"}
        else:
            lang = "Turkish" if probe == "M2b_B" else "English"
            system = (f"You are answering a factual QA question. Give the answer in "
                      f"{lang}, concise and exact.")
            rec = call_with_retries(client, model, system, surface + "\n\nCevap:",
                                    benchmark=benchmark, item=item)
            if probe == "M2b_B":
                match, lev, sim, lmatch, legacy = qa_score_robust(
                    row["cevap"], rec["content"], surface)
                gold_used = "tr"
            else:                             # arm C: score against EN gold (fix 3)
                gold_en = _ensure_gold_en(client, row, item)
                if not gold_en:
                    match, lev, sim, lmatch, legacy = False, "no-en-gold", 0.0, False, False
                else:
                    match, lev, sim, lmatch, legacy = qa_score_robust(
                        gold_en, rec["content"], surface)
                gold_used = "en"
            out = {"qa_match": match, "qa_level": lev, "qa_sim": sim,
                   "qa_match_lenient": lmatch, "qa_legacy_containment": legacy,
                   "gold_used": gold_used,
                   "answer_correct": match, "arm": "B" if probe == "M2b_B" else "C"}
        return _record(benchmark, model, probe, item, rec, out)

    raise ValueError(f"unknown probe {probe}")


def _record(benchmark: str, model: str, probe: str, item: int,
            rec: dict, out: dict) -> dict:
    usage = rec["usage"]
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    ptd = usage.get("prompt_tokens_details") or {}
    ctd = usage.get("completion_tokens_details") or {}
    cached = int(ptd.get("cached_tokens") or 0)
    reason = int(ctd.get("reasoning_tokens") or 0)
    cost = cost_estimate(model, usage)
    base = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": "full_audit_2026-08-16",
        "benchmark": benchmark,
        "model": model,
        "probe": probe,
        "probe_family": probe.split("_")[0],
        "item_idx": item,
        "item_key": f"{benchmark}|{model}|{probe}|{item}",
        "prompt_tokens": pt, "cached_tokens": cached,
        "completion_tokens": ct, "reasoning_tokens": reason,
        "total_tokens": pt + ct,
        "usage_api": usage,
        "cost_est_usd": cost,
        "retries": rec["retries"],
        "retry_log": rec.get("retry_log", []),
        "retry_hard": rec.get("retry_hard"),
        "ok": rec["ok"], "error": rec["error"], "took_s": rec["took_s"],
        "raw": rec["content"][:400],
    }
    base.update(out)
    return base


# ---------------------------------------------------------------------------
# Cost estimator: pilot-real token profile per (model, probe) -> dry-run
# ---------------------------------------------------------------------------
TOKEN_PROFILE = {   # mean (prompt, completion) tokens, pilot 2026-08-16
    "gpt-5.6-luna": {"M1": (146, 150), "M2a": (159, 54), "M2b_A": (120, 115),
                     "M2b_B": (95, 131), "M2b_C": (92, 122)},
    "mimo-v2.5": {"M1": (153, 486), "M2a": (164, 281), "M2b_A": (123, 313),
                  "M2b_B": (103, 358), "M2b_C": (94, 297)},
    "deepseek-v4-flash": {"M1": (247, 416), "M2a": (264, 384), "M2b_A": (214, 387),
                          "M2b_B": (183, 430), "M2b_C": (165, 341),
                          "M2b_tr_en": (225, 310), "M2b_en_tr": (174, 385),
                          "M2b_gold_en": (225, 310)},
}


def build_tasks(benchmarks: List[str], models: List[str], samples: Dict,
                probes: Optional[List[str]] = None) -> List[tuple]:
    tasks = []
    for b in benchmarks:
        for m in models:
            for probe in probes or ["M1", "M2a", "M2b_A", "M2b_B", "M2b_C"]:
                if probe == "M2a" and b == "ragturk_formal5k":
                    continue
                for i in range(len(samples[b])):
                    tasks.append((b, m, probe, i))
    return tasks


def estimate_cost(tasks, samples, benchmarks) -> float:
    total = 0.0
    n_per = {}
    for b, m, probe, i in tasks:
        n_per[(m, probe)] = n_per.get((m, probe), 0) + 1
    for (m, probe), n in n_per.items():
        prof = TOKEN_PROFILE.get(m, {}).get(probe)
        if prof is None or LADDER.get(m) is None:
            continue
        pt, ct = prof
        p_in, p_c, p_out = LADDER[m]
        total += n * (pt * p_in + ct * p_out) / 1e6
    n_trans = 2 * len(benchmarks) * len(samples[benchmarks[0]]) if benchmarks else 0
    dprof = TOKEN_PROFILE["deepseek-v4-flash"]
    p_in, p_c, p_out = LADDER["deepseek-v4-flash"]
    total += n_trans * (dprof["M2b_tr_en"][0] * p_in + dprof["M2b_tr_en"][1] * p_out) / 1e6
    total += n_trans * (dprof["M2b_en_tr"][0] * p_in + dprof["M2b_en_tr"][1] * p_out) / 1e6
    if "ragturk_formal5k" in benchmarks:     # gold tr->en for arm C (fix 3)
        total += len(samples["ragturk_formal5k"]) * (
            dprof["M2b_gold_en"][0] * p_in + dprof["M2b_gold_en"][1] * p_out) / 1e6
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate", type=float, default=2.0, help="global req/s")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--spend-log", default=DEFAULT_SPEND)
    ap.add_argument("--models", default=",".join(ANS_MODELS))
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--max-cost", type=float, default=2.00)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probes", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    probes = [p.strip() for p in args.probes.split(",")] if args.probes else None

    samples = {b: load_sample(b, args.limit, seed=SEED) for b in benchmarks}
    tasks = build_tasks(benchmarks, models, samples, probes)
    est = estimate_cost(tasks, samples, benchmarks)
    n_calls = len(tasks)
    n_trans = 2 * len(benchmarks) * (len(samples[benchmarks[0]]) if benchmarks else 0)
    n_golden = len(samples.get("ragturk_formal5k", [])) if "ragturk_formal5k" in benchmarks else 0

    print(f"=== FULL RUN PLAN (Day 8, 2026-08-16) ===")
    print(f"benchmarks : {benchmarks}")
    print(f"models     : {models}")
    print(f"probes     : {probes or ['M1','M2a','M2b_A','M2b_B','M2b_C']} "
          f"(+2 translation calls per item, lazy; +1 gold_TR->EN per RAGTurk item)")
    print(f"n per benchmark sample (seed 42) : { {b: len(samples[b]) for b in benchmarks} }")
    print(f"planned answer calls: {n_calls} | translations: {n_trans} | "
          f"RAGTurk gold->EN: {n_golden} | TOTAL ~{n_calls + n_trans + n_golden}")
    print(f"estimated cost (documented rates only; mimo excluded, tokens counted): "
          f"${est:.4f}  [cap ${args.max_cost:.2f}]")
    print(f"fix 2 retry-hard: {RETRY_HARD_MODEL} x {RETRY_HARD_BENCH} "
          f"(surface>{LONG_SURFACE_CHARS} chars or item-recurrence), max {RETRY_HARD_MAX} retries")
    print(f"fix 3 matcher: qa_matcher.py strict "
          f"(precision 1.000 / recall 0.371 on labeled set); arm-C scored vs EN-translated gold")
    print(f"fix 4 spend checkpoints: every 500 calls -> {os.path.basename(args.spend_log)}")
    if args.dry_run:
        print("[dry-run] no calls made. OK to proceed." if est <= args.max_cost else
              f"[dry-run] ABORT: estimate ${est:.4f} exceeds cap ${args.max_cost:.2f}.")
        return
    if est > args.max_cost:
        print(f"HARD ABORT: estimated cost ${est:.4f} exceeds --max-cost "
              f"${args.max_cost:.2f}. No calls made.")
        return 1

    # resume: skip keys already present; re-seed caches from logged records
    done = set()
    n_events = 0
    if os.path.exists(args.log):
        with open(args.log, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event"):
                    n_events += 1
                    continue
                done.add(rec.get("item_key"))
                if not rec.get("ok"):
                    if rec.get("model") == RETRY_HARD_MODEL:
                        with _FAILED_ITEMS_LOCK:
                            _FAILED_LUNA_ITEMS.add((rec.get("benchmark"), rec.get("item_idx")))
                    continue
                if rec.get("probe") in ("M2b_tr_en", "M2b_en_tr"):
                    key = (rec["benchmark"], rec["item_idx"])
                    surf = _TRANS_CACHE.setdefault(key, {})
                    if rec["probe"] == "M2b_tr_en":
                        surf["en_direct"] = rec.get("translation", "")
                    else:
                        surf["backtr_tr"] = rec.get("translation", "")
                elif rec.get("probe") == "M2b_gold_en":
                    _GOLD_EN_CACHE[rec["item_idx"]] = rec.get("translation", "")
    todo = [t for t in tasks if f"{t[0]}|{t[1]}|{t[2]}|{t[3]}" not in done]
    resumed = len(done) > 0
    print(f"resume: {len(done)} calls already logged -> skipping, {len(todo)} to run "
          f"(translation cache re-seeded: {len(_TRANS_CACHE)} items, "
          f"gold-EN cache: {len(_GOLD_EN_CACHE)}; prior events in log: {n_events})")

    global BUCKET
    BUCKET = TokenBucket(args.rate)
    MAX_COST[0] = args.max_cost
    ABORT.clear()
    CALLS_SINCE_CHECKPOINT[0] = len(done)
    LOG_PATH[0] = args.log
    SPEND_PATH[0] = args.spend_log
    client = core.make_client()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_event("run_start", {"resumed": resumed, "resume_skipped": len(done),
                            "todo": len(todo), "cost_cap_usd": args.max_cost})

    def run_one(task):
        if ABORT.is_set():
            return (0, 0, 0)
        b, m, probe, i = task
        BUCKET.acquire()
        try:
            rec = run_probe_call(client, b, m, samples[b], probe, i)
            if rec is None:
                return (0, 0, 0)            # degenerate item; never a call
            log_record(rec)
            tag = f"{b[:10]}/{m[:16]}/{probe}"
            rh = f" RH={rec.get('retry_hard')}" if rec.get("retry_hard") else ""
            print(f"[{tag} {i}] {'ok' if rec['ok'] else 'FAIL'}{rh} "
                  f"retries={rec['retries']} tok={rec['total_tokens']} "
                  f"cost≈{rec['cost_est_usd']}")
            return (int(rec["ok"]), int(not rec["ok"]), rec["retries"])
        except Exception as exc:
            print(f"[EXC] {b}|{m}|{probe}|{i}: {type(exc).__name__}: {exc}")
            return (0, 1, 0)

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    n_ok = n_fail = n_ret = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ok, fail, ret in ex.map(run_one, todo):
                n_ok += ok
                n_fail += fail
                n_ret += ret
    dt = time.time() - t0
    with COST_LOCK:
        final_cost = SPENT[0]
    n_run = n_ok + n_fail
    _checkpoint()   # final checkpoint line
    log_event("run_end", {"run_calls": n_run, "ok": n_ok, "fail": n_fail,
                          "retries": n_ret, "elapsed_s": round(dt, 1)})
    print(f"\n=== FULL RUN SUMMARY (Day 8) ===")
    print(f"planned={n_calls} run={n_run} ok={n_ok} fail={n_fail} retries={n_ret}")
    print(f"elapsed={dt:.0f}s  rate~{n_run/max(dt,1):.2f} calls/s")
    print(f"accumulated estimated cost (rated models) = ${final_cost:.4f} "
          f"[cap ${args.max_cost:.2f}]")
    print(f"log -> {args.log}")
    if ABORT.is_set():
        print("NOTE: run aborted early by live cost cap; rerun same command to resume.")


if __name__ == "__main__":
    sys.exit(main())