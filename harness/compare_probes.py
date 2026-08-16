"""Compare Black Box vs Choice-Substitution per question (same seed-42 sample)."""
import json, random, re
from collections import defaultdict
from datasets import load_dataset

BB  = r"C:\Users\oguzc\ai-team\research\harness\results\pilot.jsonl"
CS  = r"C:\Users\oguzc\ai-team\research\harness\results\pilot_choicesub.jsonl"
N   = 100

ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
rng = random.Random(42)
sample = rng.sample(ds, min(N, len(ds)))

def load(fn):
    return {json.loads(l)["i"]: json.loads(l) for l in open(fn, encoding="utf-8") if l.strip()}
bb = load(BB); cs = load(CS)

def parse_letter(raw):
    raw = (raw or "").strip()
    m = re.search(r'\b([A-Ea-e])\b\s*\)', raw)
    if m: return m.group(1).upper()
    m = re.fullmatch(r'\s*([A-Ea-e])\s*', raw)
    if m: return m.group(1).upper()
    m = re.search(r'\bCevap:\s*([A-Ea-e])\b', raw)
    if m: return m.group(1).upper()
    return None

def idx_letter(i): return chr(65 + int(i))

bb_acc = cs_acc = 0; bb_n = cs_n = 0
flip_bb_to_cs = 0;      # correct in BB, wrong/None in CS -> signal
paired = 0
for i, row in enumerate(sample):
    truth = idx_letter(row["cevap"])
    if i in bb and i in cs:
        paired += 1
        pb = parse_letter(bb[i]["raw"])
        pc = parse_letter(cs[i]["raw"])
        bb_n += (pb is not None)
        cs_n += (pc is not None)
        if pb == truth: bb_acc += 1
        if pc == truth: cs_acc += 1
        if pb == truth and pc != truth:
            flip_bb_to_cs += 1
    elif i in bb:
        pb = parse_letter(bb[i]["raw"])
        bb_n += (pb is not None)
        if pb == truth: bb_acc += 1

print(f"paired questions: {paired}")
print(f"\n=== COMPARISON (DeepSeek v4 Flash, seed 42, n={N}) ===")
print(f"Black Box         : parsed {bb_n}, correct {bb_acc}, acc = {bb_acc/bb_n:.3f}" if bb_n else "n/a")
print(f"Choice-Substitution: parsed {cs_n}, correct {cs_acc}, acc = {cs_acc/cs_n:.3f}" if cs_n else "n/a")
print(f"per-question BB->CS accuracy drop (was correct in BB, then wrong in CS): {flip_bb_to_cs}/{paired}")
# subject-level degradation hotspots
print("\nhighest-degradation subjects (BB correct but CS failed):")
def is_correct(rec, row):
    return parse_letter(rec["raw"]) == idx_letter(row["cevap"])
degr = defaultdict(list)
for i, row in enumerate(sample):
    bb_ok = i in bb and is_correct(bb[i], row)
    cs_ok = i in cs and is_correct(cs[i], row)
    if bb_ok and not cs_ok:
        degr[row["bolum"]].append(i)
for s, idxs in sorted(degr.items(), key=lambda kv: -len(kv[1]))[:6]:
    print(f"  {s}: {len(idxs)} items {idxs[:5]}")