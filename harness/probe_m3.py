"""Probe M3 — corpus-overlap statistics (contiguous 13-gram exact match). Day 9.

Design section 3 of paper1_contamination_audit.md: for every benchmark item's
surface text (question + choices), tokenize into lowercase word tokens, form
every contiguous 13-gram, hash each gram to a 64-bit integer (hashlib.md5,
truncated to 8 bytes), and intersect that item key-set with the same key-set
built from a web-scale corpus the models plausibly saw. Output metric: the
proportion of items with >=1 matching 13-gram plus per-item matched-gram
counts. This is a DESCRIPTIVE baseline that contextualizes probes M1/M2; it
does not by itself prove training inclusion (black-box limitation, stated).

Corpus sources (either or both; --dry-run loads none):
  1) Wikipedia tr dump (XML, parsed streaming with xml.etree.iterparse;
     <text> bodies of namespace-0 pages are tokenized and keyed; redirects,
     empty text and non-article namespaces are skipped):
       Dump URL:  https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2
       Example:   uv run python probe_m3.py --wiki /path/to/trwiki-latest-pages-articles.xml.bz2
       (compressed .bz2/.gz/.xz handled via stdlib bz2/gzip/lzma)
  2) CulturaX-tr via the 'datasets' package (uonlp/CulturaX, config 'tr'),
     streamed and capped at --culturax-items N (paper run cap: 10000):
       Example:   uv run python probe_m3.py --culturax-items 10000
       Both:      uv run python probe_m3.py --wiki <path> --culturax-items 10000

Memory: gram keys are 8-byte ints stored in a set (no full-gram strings kept).
A full tr-wiki pages-articles dump (~2 GB XML, ~700k articles; Wikipedia tr
has ~600k+ articles) yields on the order of 5e8 grams — approx 25-40 GB as a
Python int set — so this script honors --max-corpus-grams (default 50,000,000,
~3-5 GB RAM) and --wiki-pages-cap for smaller/smoke runs. Use --dry-run to
check benchmark loading and token stats without touching any corpus.

--dry-run loads ONLY the benchmark CSVs under data/ and prints item/token/gram
stats; it never reads corpora, never touches network, and never writes to
harness/results/ (safe to run while the n=200 inference run is executing).

Output (non-dry-run): per benchmark per corpus, one JSONL line appended to
--out (default results/m3_overlap_2026-08-16.jsonl) with items_with_match
(= count of items with >=1 matching 13-gram), match_rate, totals and a
per_item array of per-item matched-gram counts; plus a printed summary table.
"""
from __future__ import annotations

import argparse
import ast
import bz2
import csv
import gzip
import hashlib
import json
import lzma
import os
import re
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

NGRAM = 13
HASH_BYTES = 8                       # md5 truncation -> 64-bit int keys
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data")
DEFAULT_OUT = os.path.join(HERE, "results", "m3_overlap_2026-08-16.jsonl")
DEFAULT_CULTURAX = "uonlp/CulturaX"
DEFAULT_CULTURAX_CONFIG = "tr"
DEFAULT_MAX_CORPUS_GRAMS = 50_000_000

# name -> (csv filename, kind, qid column or None)
# kind "mc" = question + choices; kind "qa" = question + answer.
BENCH_CSVS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "tr_mmlu":          ("tr_mmlu.csv",          "mc", None),
    "tumlu_tr":         ("tumlu_tr.csv",         "mc", None),
    "ragturk_formal5k": ("ragturk_formal5k.csv", "qa", "qid"),
    "halluverse_tr":    ("halluverse_tr.csv",    "qa", "qid"),
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Tokenizer / gram hashing (shared by benchmark items AND corpus streams)
# ---------------------------------------------------------------------------
def word_tokens(text: str) -> List[str]:
    """Lowercased word tokens (unicode \w includes Turkic letters)."""
    return TOKEN_RE.findall(text.lower())


def _gram_hash(gram: str) -> int:
    """Deterministic 64-bit key: md5(gram) truncated to 8 bytes."""
    return int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:HASH_BYTES], "big")


def iter_gram_keys(tokens: List[str], n: int = NGRAM) -> Iterator[int]:
    """Yield the key of every contiguous n-gram of word tokens."""
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield _gram_hash(" ".join(tokens[i:i + n]))


def tokens_and_grams(text: str, n: int = NGRAM) -> Tuple[int, int]:
    toks = word_tokens(text)
    return len(toks), max(0, len(toks) - n + 1)


# ---------------------------------------------------------------------------
# Benchmark loading (normalized CSVs; schema verified 2026-08-16)
# ---------------------------------------------------------------------------
def parse_choices(raw) -> List[str]:
    """secenekler is stored as a Python list-literal string; parse robustly."""
    if isinstance(raw, list):
        return [str(c) for c in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(c) for c in val]
        return [str(val)]
    except Exception:
        # fallback: naive split (only reached on malformed reprs)
        return [c.strip().strip("'\" ") for c in s.strip("[]() ").split(",") if c.strip()]


def _item_text(row: Dict[str, str], kind: str) -> str:
    q = str(row.get("soru", "") or "").strip()
    if kind == "mc":
        choices = parse_choices(row.get("secenekler", ""))
        body = " ".join(str(c).strip() for c in choices if str(c).strip())
        return (q + " " + body).strip()
    # qa-pair: question + ground-truth answer
    a = str(row.get("cevap", "") or "").strip()
    return (q + " " + a).strip()


def load_benchmarks(data_dir: str = DEFAULT_DATA) -> Dict[str, List[Dict]]:
    """Load all benchmark items -> {benchmark: [{idx, qid?, text}]}."""
    out: Dict[str, List[Dict]] = {}
    for name, (fname, kind, qid_col) in BENCH_CSVS.items():
        path = os.path.join(data_dir, fname)
        items: List[Dict] = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for i, raw in enumerate(csv.DictReader(f)):
                row = {k: (v or "").strip() for k, v in raw.items()}
                text = _item_text(row, kind)
                item: Dict = {"idx": i, "text": text}
                if qid_col and row.get(qid_col):
                    item["qid"] = row[qid_col]
                items.append(item)
        out[name] = items
    return out


# ---------------------------------------------------------------------------
# Corpus builders  (identical tokenizer; grams hash-added to one key set)
# ---------------------------------------------------------------------------
def _smart_open_binary(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rb")
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith((".xz", ".lzma")):
        return lzma.open(path, "rb")
    return open(path, "rb")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag else ""


def build_wiki_keys(path: str, pages_cap: Optional[int] = None,
                    max_grams: int = DEFAULT_MAX_CORPUS_GRAMS,
                    n: int = NGRAM) -> Tuple[set, Dict]:
    """Stream-parse a MediaWiki XML dump; return (key_set, meta)."""
    keys: set = set()
    pages = chars = grams = redirects = skipped = 0
    truncated = False
    text_buf = ""
    ns = ""
    is_redirect = False
    with _smart_open_binary(path) as f:
        for event, elem in ET_iterparse(f):
            tag = _local(elem.tag)
            if tag == "text":
                text_buf = elem.text or ""
                elem.clear()
            elif tag == "ns":
                ns = (elem.text or "").strip()
                elem.clear()
            elif tag == "redirect":
                is_redirect = True
                elem.clear()
            elif tag == "page":
                pages += 1
                if pages_cap is not None and pages > pages_cap:
                    elem.clear()
                    break
                if ns == "0" and text_buf and not is_redirect and not text_buf.startswith(("#REDIRECT", "#YÖNLENDİR")):
                    chars += len(text_buf)
                    toks = word_tokens(text_buf)
                    for gk in iter_gram_keys(toks, n):
                        keys.add(gk)
                        grams += 1
                        if grams >= max_grams:
                            truncated = True
                            break
                elif is_redirect:
                    redirects += 1
                else:
                    skipped += 1
                elem.clear()
                text_buf, ns, is_redirect = "", "", False
                if pages % 50_000 == 0:
                    print(f"[wiki] pages={pages:,} grams={grams:,} chars={chars:,}")
                if truncated:
                    break
    meta = {
        "source": "wikipedia_tr", "path": os.path.basename(path),
        "pages": pages, "chars": chars, "redirects_skipped": redirects,
        "empty_or_other_ns_skipped": skipped, "grams": grams,
        "max_grams": max_grams, "truncated": truncated,
    }
    return keys, meta


def ET_iterparse(f):
    """lazy import to keep --dry-run stdlib-only and cheap"""
    import xml.etree.ElementTree as ET
    return ET.iterparse(f, events=("end",))


def build_culturax_keys(items_cap: int, dataset: str = DEFAULT_CULTURAX,
                        config: str = DEFAULT_CULTURAX_CONFIG,
                        max_grams: int = DEFAULT_MAX_CORPUS_GRAMS,
                        n: int = NGRAM) -> Tuple[set, Dict]:
    """Stream CulturaX-tr rows from HF (datasets), cap at items_cap."""
    keys: set = set()
    items = chars = grams = skipped = 0
    truncated = False
    from datasets import load_dataset  # heavy import kept off the dry-run path
    ds = load_dataset(dataset, config, split="train", streaming=True)
    for row in ds:
        if items >= items_cap:
            break
        text = str(row.get("text", "") or "")
        if not text.strip():
            skipped += 1
            items += 1
            continue
        items += 1
        chars += len(text)
        for gk in iter_gram_keys(word_tokens(text), n):
            keys.add(gk)
            grams += 1
            if grams >= max_grams:
                truncated = True
                break
        if (items % 1000) == 0:
            print(f"[culturax] items={items:,} grams={grams:,} chars={chars:,}")
        if truncated:
            break
    meta = {
        "source": "culturax_tr", "dataset": dataset, "config": config,
        "items": items, "chars": chars, "empty_skipped": skipped,
        "grams": grams, "max_grams": max_grams, "truncated": truncated,
    }
    return keys, meta


# ---------------------------------------------------------------------------
# Evaluation of one benchmark against one corpus key-set
# ---------------------------------------------------------------------------
def evaluate_benchmark(bench: str, items: List[Dict], keys: set,
                       n: int = NGRAM) -> Dict:
    n_items = len(items)
    hits = 0
    total_item_grams = 0
    total_matched = 0
    per_item = []
    for it in items:
        toks = word_tokens(it["text"])
        item_grams = set(iter_gram_keys(toks, n))
        matched = len(item_grams & keys)
        hit = matched > 0
        hits += int(hit)
        total_item_grams += len(item_grams)
        total_matched += matched
        rec: Dict = {"idx": it["idx"], "grams": len(item_grams),
                     "matched": matched, "hit": hit}
        if "qid" in it:
            rec["qid"] = it["qid"]
        per_item.append(rec)
    return {
        "benchmark": bench,
        "n_items": n_items,
        "items_with_match": hits,
        "match_rate": round(hits / n_items, 6) if n_items else 0.0,
        "total_item_grams": total_item_grams,
        "matched_grams": total_matched,
        "grams_matched_rate": round(total_matched / total_item_grams, 6)
                              if total_item_grams else 0.0,
        "per_item": per_item,
    }


def print_table(corpus_label: str, rows: List[Dict]) -> None:
    print(f"\n=== M3 summary — corpus: {corpus_label} ===")
    print(f"{'benchmark':<22}{'items':>8}{'match>=1':>10}{'rate%':>9}"
          f"{'item_grams':>14}{'matched':>12}")
    for r in rows:
        print(f"{r['benchmark']:<22}{r['n_items']:>8,}{r['items_with_match']:>10,}"
              f"{r['match_rate'] * 100:>8.2f}%{r['total_item_grams']:>14,}"
              f"{r['matched_grams']:>12,}")


# ---------------------------------------------------------------------------
# Dry run: benchmark item counts + token stats only (no corpus, no network,
# no writes under results/)
# ---------------------------------------------------------------------------
def dry_run(items: Dict[str, List[Dict]], n: int) -> int:
    print("=== M3 dry-run: benchmark items + token stats (no corpus loaded) ===")
    print(f"{'benchmark':<22}{'items':>8}{'tokens':>12}{'grams13':>12}"
          f"{'mean_tok':>10}{'mean_gram':>10}")
    total_items = total_toks = total_grams = 0
    for bench, rows in items.items():
        toks = gs = 0
        for it in rows:
            t, g = tokens_and_grams(it["text"], n)
            toks += t
            gs += g
        total_items += len(rows)
        total_toks += toks
        total_grams += gs
        print(f"{bench:<22}{len(rows):>8,}{toks:>12,}{gs:>12,}"
              f"{toks / len(rows):>10.1f}{gs / len(rows):>10.1f}")
    print(f"\ntotals: items={total_items:,} word_tokens={total_toks:,} "
          f"13-grams={total_grams:,}")
    print("(per item, 'question + choices' for MC benchmarks, "
          "'question + answer' for QA-pair benchmarks)")
    print("corpus run (non-dry):  uv run python probe_m3.py --wiki <path>  "
          "and/or  uv run python probe_m3.py --culturax-items 10000")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Probe M3: 13-gram corpus-overlap of benchmark item text "
                    "against Wikipedia tr / CulturaX-tr (design section 3).")
    ap.add_argument("--wiki", metavar="PATH", default=None,
                    help="MediaWiki XML dump file (plain or .bz2/.gz/.xz), "
                         "e.g. https://dumps.wikimedia.org/trwiki/latest/"
                         "trwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--culturax-items", metavar="N", type=int, default=None,
                    help="enable the CulturaX-tr corpus via 'datasets' "
                         "(uonlp/CulturaX, config 'tr'), capped at N streamed "
                         "items (paper run cap: 10000; 0 forces off)")
    ap.add_argument("--culturax-dataset", default=DEFAULT_CULTURAX,
                    help=f"HF dataset id (default {DEFAULT_CULTURAX})")
    ap.add_argument("--culturax-config", default=DEFAULT_CULTURAX_CONFIG,
                    help=f"HF dataset config (default {DEFAULT_CULTURAX_CONFIG})")
    ap.add_argument("--ngram", type=int, default=NGRAM,
                    help=f"contiguous word-token gram size (default {NGRAM})")
    ap.add_argument("--max-corpus-grams", type=int, default=DEFAULT_MAX_CORPUS_GRAMS,
                    help=f"hard cap on grams hashed into the corpus key set "
                         f"(default {DEFAULT_MAX_CORPUS_GRAMS:,}; ~3-5 GB RAM)")
    ap.add_argument("--wiki-pages-cap", type=int, default=None,
                    help="optional early-stop after N wiki pages (smoke runs)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="benchmark CSV dir")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="JSONL output path (default results/"
                         "m3_overlap_2026-08-16.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only load benchmarks + print item/token/gram stats; "
                         "no corpus, no network, no results writes")
    args = ap.parse_args()

    items = load_benchmarks(args.data)

    if args.dry_run:
        return dry_run(items, args.ngram)

    if not args.wiki and not (args.culturax_items and args.culturax_items > 0):
        ap.error("specify at least one corpus source: --wiki PATH "
                 "and/or --culturax-items N (or use --dry-run)")

    corpora: List[Tuple[str, set, Dict]] = []
    if args.wiki:
        print(f"[wiki] streaming {args.wiki} ...")
        keys, meta = build_wiki_keys(args.wiki, pages_cap=args.wiki_pages_cap,
                                     max_grams=args.max_corpus_grams, n=args.ngram)
        corpora.append(("wikipedia_tr", keys, meta))
    if args.culturax_items and args.culturax_items > 0:
        print(f"[culturax] streaming {args.culturax_dataset} "
              f"config={args.culturax_config} cap={args.culturax_items} ...")
        keys, meta = build_culturax_keys(args.culturax_items,
                                         dataset=args.culturax_dataset,
                                         config=args.culturax_config,
                                         max_grams=args.max_corpus_grams,
                                         n=args.ngram)
        corpora.append(("culturax_tr", keys, meta))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for corpus_name, keys, meta in corpora:
            rows: List[Dict] = []
            for bench, bench_items in items.items():
                rec = evaluate_benchmark(bench, bench_items, keys, args.ngram)
                rec.update({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "probe": f"M3-{args.ngram}gram",
                    "corpus": corpus_name,
                    "corpus_meta": meta,
                })
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(rec)
            print_table(f"{corpus_name} (grams={meta['grams']:,}, "
                        f"truncated={meta['truncated']})", rows)
    print(f"\nwrote per-benchmark JSONL lines + per-item match counts to "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())