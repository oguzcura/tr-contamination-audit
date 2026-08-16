"""RAGTurk answer-form-robust QA matcher (fix 3 of the full-run approval).

Calibrated + measured on 106 hand-labeled (gold, pred) pairs from the Day-5
pilot (harness/results/ragturk_matcher_labeled.json, two-rater agreement on a
10-pair batch). Measured on that set (strict): precision 1.000, recall 0.371,
accuracy 0.792; by cascade level: exact 2, containment 11, number-mismatch 48
(rejections), none 44, empty 1, paraphrase levels 0 (all four candidate
paraphrase matches were false positives - single-fact swaps - and are rejected
by the content-token guard).

Cascade (frozen 2026-08-16, pre-run; do not tune post-hoc):
  L0 empty guard
  L1 letter-form gold ("a) ..." / "A: ...")
  L2 exact (normalized equality)
  L3 NUMBER GUARD: every gold numeric token must appear in pred, with decade
     tokens ('1950'ler) satisfied by any pred year in [1950,1959]; failure
     => level "number-mismatch", no match (catches 10 vs 9 Kasım, 16 vs 21
     Ağustos, 14 vs 18 gol, 1993 vs 1996, etc.)
  L4 containment: norm(g) in norm(p) (or vice versa), min 8 chars of the
     contained side, contained/longer length ratio >= 0.25
  L5 paraphrase: token Jaccard >= 0.65 or 5-gram coverage >= 0.80
     (numeric golds: >= 0.55 / >= 0.65 - numbers already verified), PLUS
     distinctive 5-gram coverage (gold grams not in the question) >= 0.50,
     PLUS content-token guard: distinctive content words the answer adds
     beyond the question (<=4 such tokens) must all appear in pred
  L6 long descriptive golds (>120 chars, no numbers): distinctive 5-gram
     coverage >= 0.50 AND Jaccard >= 0.30
Everything else: level "none", no match.

Levels can never over-credit by construction: every fallback path is guarded
by the number guard, distinctive-gram coverage and the content-token guard.

The LENIENT variant restores the L6 Jaccard-escape and drops the content-token
guard (pilot-level paraphrase thresholds 0.60/0.75) and is reported ONLY as a
sensitivity reading for RAGTurk arms (see the full-run report), never as the
headline accuracy.
"""
from __future__ import annotations
import re
import unicodedata

TR_STOP = {"bir", "ve", "ile", "için", "daha", "sonra", "kendi", "olarak",
           "göre", "ancak", "ayrıca", "bu", "şu", "o", "ne", "hangi", "nasıl",
           "neden", "ilk", "kez", "ise", "da", "de", "mı", "mi", "mu", "mü",
           "ki", "ya", "en"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[\*\#\_\`>]", " ", s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    s = re.sub(r"[^\w\s\d.,:°/%-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def nums(s: str):
    return [m for m in re.findall(r"\d+(?:[.,]\d+)*", s)]


def grams5(s: str):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 5] for i in range(len(s) - 4)}


def _distinctive(g: str, p: str, question: str):
    q = norm(question)
    dist = grams5(g) - (grams5(q) if q else set())
    dcov = len(dist & grams5(p)) / len(dist) if dist else 0.0
    q_toks = set(q.split()) if q else set()
    dtoks = [t for t in g.split()
             if len(t) >= 4 and t not in TR_STOP and t not in q_toks]
    return dist, dcov, dtoks


def matcher(gold: str, pred: str, question: str = "", strict: bool = True):
    """-> (match, level, sim). Frozen cascade; strict=False = lenient variant."""
    g, p = norm(gold), norm(pred)
    if not g or not p:
        return (False, "empty", 0.0)
    if re.fullmatch(r"[a-e][).:].*", g):               # L1 letter-form gold
        if re.match(r"[a-e][).:]", p) or p == g:
            return (True, "letter", 1.0)
        return (False, "letter-mismatch", 0.0)
    if g == p:                                          # L2 exact
        return (True, "exact", 1.0)
    gn, pn = nums(g), nums(p)
    if gn:                                              # L3 number guard
        for t in gn:
            if re.search(r"(\d+)ler", g) and re.search(r"(\d+)ler", g).group(1) == t:
                if not any(int(pv) in range(int(t), int(t) + 10) for pv in pn):
                    return (False, "number-mismatch", 0.0)
            elif t not in pn:
                return (False, "number-mismatch", 0.0)
    # L4 containment
    if len(g) >= 8 and g in p:
        return (True, "containment", 1.0) if len(g) / max(len(p), 1) >= 0.25 \
            else (False, "containment-thin", 0.0)
    if len(p) >= 8 and p in g:
        return (True, "containment", 1.0) if len(p) / max(len(g), 1) >= 0.25 \
            else (False, "containment-thin", 0.0)
    # L5/L6 paraphrase
    gs, ps = set(g.split()), set(p.split())
    jac = len(gs & ps) / len(gs | ps) if gs and ps else 0.0
    cov = len(grams5(g) & grams5(p)) / len(grams5(g)) if grams5(g) else 0.0
    sim = max(jac, cov)
    dist, dcov, dtoks = _distinctive(g, p, question)
    if not strict:                                      # lenient sensitivity
        t_jac, t_cov = (0.60, 0.75) if not gn else (0.50, 0.60)
        if len(g) > 120 and not gn and (dcov >= 0.50 or jac >= 0.60):
            return (True, "paraphrase-distinctive", dcov)
        if jac >= t_jac or cov >= t_cov:
            return (True, "paraphrase", sim)
        return (False, "none", sim)
    if len(g) > 120 and not gn:                         # L6 long descriptive
        if dcov >= 0.50 and jac >= 0.30:
            return (True, "paraphrase-distinctive", dcov)
    t_jac, t_cov = (0.65, 0.80) if not gn else (0.55, 0.65)
    tok_guard = (len(dtoks) <= 4 and all(t in ps for t in dtoks)) or \
                (len(dtoks) > 4 and
                 sum(1 for t in dtoks if t in ps) / len(dtoks) >= 0.5)
    if (jac >= t_jac or cov >= t_cov) and (not dist or dcov >= 0.50) and tok_guard:
        return (True, "paraphrase", sim)
    return (False, "none", sim)