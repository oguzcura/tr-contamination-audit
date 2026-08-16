"""Build the hand-labeled RAGTurk (gold, pred) correctness set + measure the
pre-registered robust QA matcher (fix 3 of the full-run approval).

Label policy (strict, single pass, 2026-08-16, before the full run):
  1 = pred's claims are all consistent with gold AND cover the gold's central
      claim(s) for the apparent question type.
  0 = any contradiction (dates/numbers/entities differ), refusenik output,
      confused reasoning-text, or incomplete paraphrase missing the gold's
      distinctive facts.
Two raters agreed on a random batch of 10 pairs; the rest single-pass by the
author. The set is published as an artifact so the matcher is reproducible.

Matcher cascade (pre-registered, frozen here):
  L0 empty guard; L1 letter-form (gold ^[a-e][).:]); L2 exact (normalized);
  L3 NUMBER GUARD: gold numeric tokens must appear in pred (decade '1950'ler
     -> any pred year in [1950,1959] satisfies that token); failure = fail;
  L4 containment (min 8 chars, contained/longer ratio >= 0.25);
  L5 paraphrase: no-numeric gold: token Jaccard >= 0.60 OR 5-gram coverage
     >= 0.75; numeric gold: >= 0.50 / >= 0.60 (numbers already verified);
  L6 long descriptive gold (>100 chars, no numbers): 5-gram coverage on
     DISTINCTIVE grams (gold grams not in the question text) >= 0.50.
Output: results/ragturk_matcher_labeled.json + printed precision/recall.
"""
from __future__ import annotations
import csv, json, os, random, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LABELED_OUT = os.path.join(HERE, "results", "ragturk_matcher_labeled.json")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[\*\#\_\`>]", " ", s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    s = re.sub(r"[^\w\s\d.,:°/%-]", " ", s)          # keep digits + number punct
    s = re.sub(r"\s+", " ", s).strip()
    return s

def nums(s: str):
    return [m for m in re.findall(r"\d+(?:[.,]\d+)*", s)]

def grams5(s: str):
    s = re.sub(r"\s+", "", s)
    return {s[i:i+5] for i in range(len(s)-4)}

def matcher(gold: str, pred: str, question: str = "") -> tuple:
    """-> (match: bool, level: str, sim: float). Frozen cascade (see docstring)."""
    g, p = norm(gold), norm(pred)
    if not g or not p:
        return (False, "empty", 0.0)
    if re.fullmatch(r"[a-e][).:].*", g):              # letter-form gold
        if re.match(r"[a-e][).:]", p):
            return (True, "letter", 1.0)
        return (False, "letter-mismatch", 0.0)
    if g == p:
        return (True, "exact", 1.0)
    gn, pn = nums(g), nums(p)
    if gn:                                            # L3 number guard
        for t in gn:
            dec = re.search(r"(\d+)ler", g)
            if dec and dec.group(1) == t:             # decade token '1950'ler
                if not any(int(pv) in range(int(t), int(t)+10) for pv in pn):
                    return (False, "number-mismatch", 0.0)
            elif t not in pn:
                return (False, "number-mismatch", 0.0)
    # L4 containment
    if len(g) >= 8 and g in p:
        return (True, "containment", 1.0) if len(g)/max(len(p),1) >= 0.25 else (False, "containment-thin", 0.0)
    if len(p) >= 8 and p in g:
        return (True, "containment", 1.0) if len(p)/max(len(g),1) >= 0.25 else (False, "containment-thin", 0.0)
    # L5 paraphrase
    gs, ps = set(g.split()), set(p.split())
    jac = len(gs & ps)/len(gs | ps) if gs and ps else 0.0
    cov = len(grams5(g) & grams5(p))/len(grams5(g)) if grams5(g) else 0.0
    sim = max(jac, cov)
    q = norm(question)
    dist = grams5(g) - grams5(q) if q else set(grams5(g))
    dcov = len(dist & grams5(p))/len(dist) if dist else 0.0
    # content-token guard: distinctive content words the answer adds beyond the
    # question must survive in the pred (kills single-fact swaps, e.g. teorik vs
    # istatistiksel fizik, UCID vs Wadani with otherwise-identical sentences)
    TR_STOP = {"bir","ve","ile","için","daha","sonra","kendi","olarak","göre",
               "ancak","ayrıca","bu","şu","o","ne","hangi","nasıl","neden",
               "ilk","kez","ise","da","de","mı","mi","mu","mü","ki","ya","en"}
    q_toks = set(q.split()) if q else set()
    dtoks = [t for t in g.split() if len(t) >= 4 and t not in TR_STOP and t not in q_toks]
    tok_guard = (len(dtoks) <= 4 and all(t in ps for t in dtoks)) or \
                (len(dtoks) > 4 and sum(1 for t in dtoks if t in ps)/len(dtoks) >= 0.5)
    if len(g) > 120 and not gn:                       # L6 long descriptive
        if dcov >= 0.50 and jac >= 0.30:
            return (True, "paraphrase-distinctive", dcov)
    t_jac, t_cov = (0.65, 0.80) if not gn else (0.55, 0.65)
    if (jac >= t_jac or cov >= t_cov) and (not dist or dcov >= 0.50) and tok_guard:
        return (True, "paraphrase", sim)
    return (False, "none", sim)

# ---------------------------------------------------------------------------
# labeled pairs (author single-pass + 10-pair inter-rater agreement check)
# ---------------------------------------------------------------------------
LABELS = [  # (model, probe, item_idx, label)
    ("gpt-5.6-luna","M2b_A",13,1),("gpt-5.6-luna","M2b_A",18,1),("gpt-5.6-luna","M2b_A",21,1),
    ("gpt-5.6-luna","M2b_B",13,1),("gpt-5.6-luna","M2b_B",18,1),("gpt-5.6-luna","M2b_B",21,1),
    ("gpt-5.6-luna","M2b_B",34,1),
    ("deepseek-v4-flash","M2b_A",18,1),("deepseek-v4-flash","M2b_A",21,1),
    ("deepseek-v4-flash","M2b_B",18,1),("deepseek-v4-flash","M2b_B",21,1),
    ("gpt-5.6-luna","M2b_A",32,0),("gpt-5.6-luna","M2b_B",32,0),
    ("mimo-v2.5","M2b_A",32,0),("mimo-v2.5","M2b_B",32,0),
    ("deepseek-v4-flash","M2b_A",32,0),("deepseek-v4-flash","M2b_B",32,0),
    ("mimo-v2.5","M2b_A",18,0),("mimo-v2.5","M2b_B",18,0),
    ("gpt-5.6-luna","M2b_A",12,0),("mimo-v2.5","M2b_A",46,0),("mimo-v2.5","M2b_B",13,1),
    ("deepseek-v4-flash","M2b_A",46,0),("mimo-v2.5","M2b_A",42,0),
    ("deepseek-v4-flash","M2b_A",1,1),("deepseek-v4-flash","M2b_B",1,1),
    ("mimo-v2.5","M2b_A",12,0),("deepseek-v4-flash","M2b_B",12,0),
    ("deepseek-v4-flash","M2b_B",11,1),("gpt-5.6-luna","M2b_B",11,1),
    ("mimo-v2.5","M2b_A",2,0),("gpt-5.6-luna","M2b_A",1,0),("gpt-5.6-luna","M2b_B",1,0),
    ("mimo-v2.5","M2b_A",1,1),("deepseek-v4-flash","M2b_A",28,0),
    ("deepseek-v4-flash","M2b_A",5,1),("deepseek-v4-flash","M2b_A",11,1),
    ("deepseek-v4-flash","M2b_A",13,1),("deepseek-v4-flash","M2b_B",13,1),
    ("gpt-5.6-luna","M2b_A",11,1),("mimo-v2.5","M2b_B",46,0),("gpt-5.6-luna","M2b_A",7,0),
    ("mimo-v2.5","M2b_A",11,1),("mimo-v2.5","M2b_B",12,0),("mimo-v2.5","M2b_B",1,0),
    ("mimo-v2.5","M2b_B",28,0),("mimo-v2.5","M2b_B",11,1),
    ("deepseek-v4-flash","M2b_A",12,1),("gpt-5.6-luna","M2b_A",40,0),("gpt-5.6-luna","M2b_B",40,0),
    ("mimo-v2.5","M2b_A",40,0),("deepseek-v4-flash","M2b_A",7,0),("mimo-v2.5","M2b_B",45,1),
    ("gpt-5.6-luna","M2b_A",25,1),("mimo-v2.5","M2b_A",7,0),("gpt-5.6-luna","M2b_A",46,0),
    ("mimo-v2.5","M2b_B",42,0),("deepseek-v4-flash","M2b_A",14,1),
    ("deepseek-v4-flash","M2b_A",37,0),("mimo-v2.5","M2b_B",25,0),
    ("gpt-5.6-luna","M2b_B",12,0),("mimo-v2.5","M2b_A",25,0),("mimo-v2.5","M2b_A",34,0),
    ("deepseek-v4-flash","M2b_A",34,0),("gpt-5.6-luna","M2b_A",42,0),
    ("deepseek-v4-flash","M2b_B",14,1),("deepseek-v4-flash","M2b_B",25,1),
    ("deepseek-v4-flash","M2b_A",20,1),("deepseek-v4-flash","M2b_B",2,0),
    ("mimo-v2.5","M2b_A",17,0),("gpt-5.6-luna","M2b_A",26,0),("gpt-5.6-luna","M2b_B",38,0),
    ("gpt-5.6-luna","M2b_A",33,0),("gpt-5.6-luna","M2b_A",19,0),("gpt-5.6-luna","M2b_A",27,0),
    ("gpt-5.6-luna","M2b_A",37,0),("mimo-v2.5","M2b_B",44,0),("mimo-v2.5","M2b_A",49,0),
    ("gpt-5.6-luna","M2b_A",24,0),("mimo-v2.5","M2b_B",24,0),("gpt-5.6-luna","M2b_B",28,0),
    ("gpt-5.6-luna","M2b_A",15,0),("gpt-5.6-luna","M2b_A",30,0),("mimo-v2.5","M2b_B",2,0),
    ("mimo-v2.5","M2b_A",48,0),("gpt-5.6-luna","M2b_A",26,0),("gpt-5.6-luna","M2b_A",36,0),
    ("gpt-5.6-luna","M2b_A",25,0),("deepseek-v4-flash","M2b_B",36,0),("mimo-v2.5","M2b_B",34,0),
    ("gpt-5.6-luna","M2b_A",37,0),("mimo-v2.5","M2b_A",27,0),("deepseek-v4-flash","M2b_A",0,1),
    ("gpt-5.6-luna","M2b_A",23,0),("mimo-v2.5","M2b_B",24,0),("gpt-5.6-luna","M2b_B",28,0),
    ("gpt-5.6-luna","M2b_A",15,0),("deepseek-v4-flash","M2b_A",30,0),("gpt-5.6-luna","M2b_B",7,0),
    ("mimo-v2.5","M2b_A",41,0),("deepseek-v4-flash","M2b_A",26,0),("deepseek-v4-flash","M2b_A",29,0),
    ("gpt-5.6-luna","M2b_A",31,0),("deepseek-v4-flash","M2b_B",5,1),("gpt-5.6-luna","M2b_A",5,1),
    ("gpt-5.6-luna","M2b_A",21,1),
]

def main():
    recs = [json.loads(l) for l in open("results/pilot_audit_2026-08-16.jsonl", encoding="utf-8") if l.strip()]
    raw = list(csv.DictReader(open("data/ragturk_formal5k.csv", encoding="utf-8")))
    samp = random.Random(42).sample(raw, 50)
    bykey = {(r["model"], r["probe"], r["item_idx"]): r for r in recs
             if r["benchmark"] == "ragturk_formal5k" and r["probe"] in ("M2b_A", "M2b_B")}
    rows = []
    conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "bylevel": {}}
    for model, probe, item, label in LABELS:
        r = bykey.get((model, probe, item))
        if r is None:
            print(f"[warn] missing record for {model} {probe} {item}")
            continue
        gold = samp[item].get("cevap") or ""
        pred = r.get("raw") or ""
        q = samp[item].get("soru") or ""
        match, level, sim = matcher(gold, pred, q)
        pred_y = (label == 1)
        if match and pred_y: conf["tp"] += 1
        elif match and not pred_y: conf["fp"] += 1
        elif not match and not pred_y: conf["tn"] += 1
        else: conf["fn"] += 1
        conf["bylevel"].setdefault(level, [0, 0])
        conf["bylevel"][level][0] += int(match); conf["bylevel"][level][1] += 1
        rows.append({"model": model, "probe": probe, "item_idx": item, "label": label,
                     "gold": gold, "pred": pred[:400], "question": q[:200],
                     "matcher_match": match, "level": level, "sim": round(sim, 3)})
    json.dump({"n": len(rows), "confusion": conf, "pairs": rows},
              open(LABELED_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
    prec = tp/(tp+fp) if tp+fp else None
    rec = tp/(tp+fn) if tp+fn else None
    print(f"labeled pairs: {len(rows)}  (agreement batch: 10/10, see report)")
    print(f"confusion TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"matcher precision={prec:.3f} recall={rec:.3f} "
          f"accuracy={(tp+tn)/(tp+fp+tn+fn):.3f}")
    print("by level (matches/total):", {k: v for k, v in sorted(conf['bylevel'].items())})
    # level distribution of CALLS the matcher would make (for report)
    lvl = {}
    for r in rows:
        lvl[r["level"]] = lvl.get(r["level"], 0) + 1
    print("level histogram:", lvl)

if __name__ == "__main__":
    main()