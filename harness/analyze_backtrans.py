"""Analyze cross-lingual probe (B=back-tr, C=en-direct) vs 100-item baseline.

Because the baseline (pilot.jsonl) is seed 42 with limit 100 and this run is seed 42
with limit 30, the FIRST 30 items are the same indices. We score B and C against
those 30 items' ground truth and compare to baseline for the paired subset.
"""
import json, re, random
from datasets import load_dataset

BASE   = r"C:\Users\oguzc\ai-team\research\harness\results\pilot.jsonl"          # 100
BC     = r"C:\Users\oguzc\ai-team\research\harness\results\pilot_backtrans_bc.jsonl"  # 30

ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
rng = random.Random(42)
sample100 = rng.sample(ds, min(100, len(ds)))

def load(fn):  # record by idx
    d={}
    for l in open(fn, encoding="utf-8"):
        l=l.strip()
        if l: r=json.loads(l); d[r["i"]]=r
    return d

bb = load(BASE); bc = load(BC)

def idx_letter(i): return chr(65 + int(i))

def parse_letter(raw):
    if not raw: return None
    raw=raw.strip()
    m=re.search(r'\b([A-Ea-e])\b\s*\)', raw)   # "D) Kontrol"
    if m: return m.group(1).upper()
    m=re.fullmatch(r'\s*([A-Ea-e])\s*', raw)   # bare "D"
    if m: return m.group(1).upper()
    m=re.search(r'\bCevap:\s*([A-Ea-e])\b', raw)  # "Cevap: C"
    if m: return m.group(1).upper()
    m=re.search(r'\b([A-Ea-e])\b', raw)          # first letter somehow
    if m: return m.group(1).upper()
    return None

# ---- baseline on the SAME 30 indices --------------------------------------
items = sorted(set(bb) & set(bc))[:30]   # paired indices present in both
print(f"paired items compared: {len(items)}")

def acc_for(rec, field):
    corr=parsed=0
    for i in items:
        truth=idx_letter(sample100[i]["cevap"])
        p=parse_letter(rec.get(i,{}).get(field))
        if p is None: continue
        parsed+=1
        if p==truth: corr+=1
    return corr, parsed, (corr/parsed if parsed else None)

bb_c, bb_p, bb_a = acc_for(bb, "raw")
b_c,  b_p,  b_a  = acc_for(bc, "B_answer")
c_c,  c_p,  c_a  = acc_for(bc, "C_answer")

print(f"\n=== 3-ARM COMPARISON (DeepSeek v4 Flash, paired seed-42 items) ===")
print(f"A baseline (original TR)  : {bb_c}/{bb_p} = {bb_a:.3f}" if bb_a else "n/a")
print(f"B back-translated TR      : {b_c}/{b_p} = {b_a:.3f}" if b_a else "n/a   (few answers parsed)")
print(f"C English-direct          : {c_c}/{c_p} = {c_a:.3f}" if c_a else "n/a")
if bb_a and b_a: print(f"  Delta A->B = {(bb_a-b_a):+.3f}  (negative = accuracy LOST under surface perturbation)")
if bb_a and c_a: print(f"  Delta A->C = {(bb_a-c_a):+.3f}  (negative = worse when moved to English)")

# ---- per-question flips ----------------------------------------------------
import collections
flip_AB=[(i, sample100[i]['bolum']) for i in items
         if parse_letter(bb.get(i,{}).get("raw"))==idx_letter(sample100[i]["cevap"])
         and parse_letter(bc.get(i,{}).get("B_answer"))!=idx_letter(sample100[i]["cevap"])]
flip_AC=[i for i in items
         if parse_letter(bb.get(i,{}).get("raw"))==idx_letter(sample100[i]["cevap"])
         and parse_letter(bc.get(i,{}).get("C_answer"))!=idx_letter(sample100[i]["cevap"])]
print(f"\nper-question baseline-correct then WRONG:")
print(f"  in B (back-translated): {len(flip_AB)}")
print(f"  in C (english-direct) : {len(flip_AC)}")

# missing answers per arm
def missing(field):
    return sum(1 for i in items if parse_letter(bc.get(i,{}).get(field)) is None)
print(f"unparseable answers: B={missing('B_answer')}, C={missing('C_answer')}")

# ---- sample the round-trip quality for native spot-check log --------------
print("\n--- native spot-check log (round-trip fidelity) ---")
for i in items[:5]:
    r=bc[i]
    print(f"\n#{i} [{r['bolum']}]  truth={idx_letter(sample100[i]['cevap'])}")
    print("  TR orig :", sample100[i]["soru"][:80].replace("\n"," "))
    print("  B rtrn  :", r.get("B_backtr","")[:80].replace("\n"," ") if r.get("B_backtr") else "(none)")
    print("  C en    :", r.get("C_en","")[:80].replace("\n"," ") if r.get("C_en") else "(none)")