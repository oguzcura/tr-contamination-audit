"""Unit check of analyze_pilot_audit stats on a synthetic log (no API spend)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def mk(bench, model, probe, item, **kw):
    d = {"benchmark": bench, "model": model, "probe": probe, "item_idx": item,
         "item_key": f"{bench}|{model}|{probe}|{item}", "ok": True, "retries": 0,
         "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "cost_est_usd": 0.0001, "usage_api": {}, "error": None,
         "raw": "A) some answer text"}
    d.update(kw)
    return d


recs = []
for i in range(10):
    recs.append(mk("tr_mmlu", "gpt-5.6-luna", "M1", i, overlap8=0.8, verbatim_hit=True))
for i in range(10):
    recs.append(mk("tr_mmlu", "gpt-5.6-luna", "M2b_A", i,
                   parsed_letter="A", answer_correct=(i < 8)))
    recs.append(mk("tr_mmlu", "gpt-5.6-luna", "M2a", i,
                   parsed_letter="A", answer_correct=(i < 5), flip_follow=(5 <= i < 8)))
    recs.append(mk("tr_mmlu", "gpt-5.6-luna", "M2b_B", i,
                   parsed_letter="A", answer_correct=(i < 4)))
    recs.append(mk("tr_mmlu", "gpt-5.6-luna", "M2b_C", i,
                   parsed_letter="A", answer_correct=(i < 8)))
with open(os.path.join(HERE, "results", "_synth.jsonl"), "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

src = open(os.path.join(HERE, "analyze_pilot_audit.py"), encoding="utf-8").read()
src = src.replace('IN = os.path.join(HERE, "results", "pilot_audit_2026-08-16.jsonl")',
                  'IN = os.path.join(HERE, "results", "_synth.jsonl")')
src = src.replace('OUT_JSON = os.path.join(HERE, "results", "pilot_stats_2026-08-16.json")',
                  'OUT_JSON = os.path.join(HERE, "results", "_synth_stats.json")')
src = src.replace('for bench in BENCHMARKS:', 'for bench in ["tr_mmlu"]:')
src = src.replace('for model in MODELS:', 'for model in ["gpt-5.6-luna"]:')
ns = {"__file__": os.path.join(HERE, "analyze_pilot_audit.py")}
exec(compile(src, "synth", "exec"), ns)
ns["main"]()

d = json.load(open(os.path.join(HERE, "results", "_synth_stats.json"), encoding="utf-8"))
c = d["cells"]["tr_mmlu|gpt-5.6-luna"]
print("M1:", c["M1"])
print("M2a flip:", c["M2a"]["flip_follow_rate"], "CI", c["M2a"]["flip_follow_ci95"])
print("M2a mcnemar:", c["M2a"]["mcnemar_discordant"], "p", c["M2a"]["mcnemar_p_raw"],
      "bonf", c["M2a"]["mcnemar_p_bonf"])
print("M2b arms:", {k: v["acc"] for k, v in c["M2b"]["arms"].items()})
print("M2b mcnemar:", {k: (v["discordant"], v["p_raw"]) for k, v in c["M2b"]["mcnemar"].items()})
print("consensus:", c["consensus_call"], "ind", c["modality_indications"])

# hand-checks
from math import sqrt
from scipy import stats as st


def wilson(k, n):
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    cc = (p + z * z / (2 * n)) / denom
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(cc - h, 0.0), 4), round(min(cc + h, 1.0), 4)


assert c["M1"]["verbatim_rate"] == 1.0 and c["M1"]["mean_overlap8"] == 0.8, "M1"
assert c["M2a"]["flip_follow_rate"] == 0.3, "M2a flip rate"
assert c["M2a"]["flip_follow_ci95"] == list(wilson(3, 10)), "M2a CI"
# McNemar M2a: base ok (i<8) & M2a wrong (i>=5): i=5,6,7 -> 3 ; base wrong (8,9) & M2a ok: 0
assert c["M2a"]["mcnemar_discordant"] == [3, 0], "M2a discordant"
assert c["M2a"]["mcnemar_p_raw"] == round(float(st.binomtest(0, 3).pvalue), 4), "M2a p"
# B vs A: base ok 0-7, B wrong i>=4 -> pos=4, neg=0
assert c["M2b"]["mcnemar"]["B_vs_A"]["discordant"] == [4, 0], "M2b discordant"
# Bonferroni family: (tr_mmlu, M2b) has 3 contrasts x 1 model = 3 tests; raw p=binomtest(0,4)
raw = round(float(st.binomtest(0, 4).pvalue), 4)
assert c["M2b"]["mcnemar"]["B_vs_A"]["p_bonf"] == round(min(1.0, raw * 3), 4), "bonf"
print("ALL HAND-CHECKS PASS")
os.remove(os.path.join(HERE, "results", "_synth.jsonl"))
os.remove(os.path.join(HERE, "results", "_synth_stats.json"))
