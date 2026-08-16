"""Analyze pilot.jsonl against TR-MMLU ground truth (seed=42 sample)."""
import json, random, re
from datasets import load_dataset

OUT = r"C:\Users\oguzc\ai-team\research\harness\results\pilot.jsonl"
N = 100

# Reload same sample the harness used (seed 42, limit 100)
ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
rng = random.Random(42)
sample = rng.sample(ds, min(N, len(ds)))

recs = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
print(f"records: {len(recs)} (expect ~{N})")

def idx_to_letter(idx):
    """TR-MMLU stores 'cevap' as an integer index into secenekler. Convert
    to the letter ('A'..'E') the model is asked to output."""
    try:
        return chr(65 + int(idx))
    except (TypeError, ValueError):
        return None

def parse_letter(raw):
    raw = (raw or "").strip()
    m = re.search(r'\b([A-Ea-e])\b\s*\)', raw)          # "C) ..."
    if m: return m.group(1).upper()
    m = re.fullmatch(r'\s*([A-Ea-e])\s*', raw)          # bare "C"
    if m: return m.group(1).upper()
    m = re.search(r'\bCevap:\s*([A-Ea-e])\b', raw)      # "Cevap: C"
    if m: return m.group(1).upper()
    return None

from collections import Counter
correct = 0; parsed = 0; noans = 0; letters = []
mismatch_examples = []
true_dist = Counter()
for rec in recs:
    row = sample[rec["i"]]
    true_idx = int(row["cevap"])           # ground truth = index into secenekler
    true_letter = idx_to_letter(true_idx)
    true_dist[true_letter] += 1
    pred = parse_letter(rec["raw"])
    if pred is None:
        noans += 1
        letters.append(None)
        continue
    letters.append(pred); parsed += 1
    if pred == true_letter:
        correct += 1
    else:
        if len(mismatch_examples) < 6:
            mismatch_examples.append((rec["i"], row["bolum"], true_letter, pred, rec["raw"][:70]))

total = len(recs)
print(f"\n=== BLACK BOX ACCURACY (seed 42, n={total}) ===")
print(f"parsed answers : {parsed}/{total}")
print(f"unparseable    : {noans}")
print(f"correct        : {correct}")
print(f"accuracy       : {correct}/{parsed} = {correct/parsed:.3f}" if parsed else "n/a")
print(f"\nresponse-letter distribution: {dict(Counter(letters))}")
print(f"ground-truth letter dist     : {dict(true_dist)}")
print(f"\n=== sample mismatches ===")
for i, subj, t, p, raw in mismatch_examples:
    print(f"  #{i} [{subj}] truth={t} pred={p} :: {raw}")

# Common-answer bias check (if model always picks one letter -> memorization/low quality)
most_common = letters and Counter([l for l in letters if l]).most_common(1)[0]
if most_common:
    print(f"\nmost common predicted letter: {most_common} "
          f"({most_common[1]/parsed:.0%} of parsed)")