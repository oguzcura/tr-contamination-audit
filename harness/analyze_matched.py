"""Matched 3-arm comparison: only items parsed in ALL of BB / B / C."""
import json, re, random
from datasets import load_dataset

BASE = r"C:\Users\oguzc\ai-team\research\harness\results\pilot.jsonl"               # 100
BC   = r"C:\Users\oguzc\ai-team\research\harness\results\pilot_backtrans_bc.jsonl"  # 30

ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
rng = random.Random(42); sample100 = rng.sample(ds, min(100, len(ds)))

def load(fn):
    d = {}
    for l in open(fn, encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l); d[r["i"]] = r
    return d
bb = load(BASE); bc = load(BC)

def parse_letter(raw):
    if not raw: return None
    raw = raw.strip()
    m = re.search(r'\b([A-Ea-e])\b\s*\)', raw)
    if m: return m.group(1).upper()
    m = re.fullmatch(r'\s*([A-Ea-e])\s*', raw)
    if m: return m.group(1).upper()
    m = re.search(r'\bCevap:\s*([A-Ea-e])\b', raw)
    if m: return m.group(1).upper()
    m = re.search(r'\b([A-Ea-e])\b', raw)
    if m: return m.group(1).upper()
    return None

def il(i): return chr(65 + int(i))

matched = []
for i in sorted(set(bb) & set(bc)):
    truth = il(sample100[i]["cevap"])
    pa, pb, pc = parse_letter(bb[i]["raw"]), parse_letter(bc[i].get("B_answer")), parse_letter(bc[i].get("C_answer"))
    if pa is None or pb is None or pc is None:
        continue
    matched.append((i, truth, pa == truth, pb == truth, pc == truth))

n = len(matched)
print(f"matched items (parsed in ALL 3 arms): {n}")
A = sum(x[2] for x in matched); B = sum(x[3] for x in matched); C = sum(x[4] for x in matched)
print(f"\n=== MATCHED 3-ARM (DeepSeek v4 Flash, seed42, n={n}) ===")
print(f"A baseline(orig TR): {A}/{n} = {A/n:.3f}")
print(f"B back-translated  : {B}/{n} = {B/n:.3f}   Delta A->B = {(A-B)/n:+.3f}")
print(f"C english-direct   : {C}/{n} = {C/n:.3f}   Delta A->C = {(A-C)/n:+.3f}")

from collections import Counter
AB = Counter((x[2], x[3]) for x in matched)
AC = Counter((x[2], x[4]) for x in matched)
def mcnemar(cnt, label):
    b = cnt[(True, False)]; c_ = cnt[(False, True)]
    return f"{label}: (A correct, then wrong)={b}   (A wrong, then correct)={c_}   -> net {b - c_:+d}"
print("\nMcNemar directional flip counts:")
print("   ", mcnemar(AB, "B back-translated"))
print("   ", mcnemar(AC, "C english-direct"))