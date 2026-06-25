#!/usr/bin/env python
"""
vr_08_specificity.py -- does value-recall measure MEMORISATION or competent estimation?

Reviewer critique (correct): the wide multiplicative distractors {0.5,0.7,1.4,2.0}x are
eliminable by order-of-magnitude, and the MCQ over-states free recall. Two added probes,
one GPU pass per model:

  WIDE  : counterfactual top-1 with {0.5,0.7,1.4,2.0}x distractors (original).
  TIGHT : counterfactual top-1 with log-matched {0.85,0.93,1.08,1.18}x distractors that
          CANNOT be eliminated by magnitude -> survives only if the model knows the value
          precisely (genuine memorisation), collapses to chance if it was estimation.
  GEN   : free greedy generation of the value (no options shown); scored by |pred-true|/true
          within 5/10/20% tolerance. The MCQ - GEN gap = how much recognition over-states recall.

Run under conda `base` (GPU). Output: outputs/leakage/specificity.{parquet,json}
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402
from vr_02_value_recall import NAMES, FY_LO, FY_HI, load_facts, discover_chrono, discover_frontier  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"
WIDE = [0.5, 0.7, 1.4, 2.0]
TIGHT = [0.85, 0.93, 1.08, 1.18]          # log-matched, magnitude-uninformative


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def candidates(rev_b, mults):
    t = fmt(rev_b)
    d = [x for x in sorted({fmt(rev_b * m) for m in mults}) if x != t and x >= 0.1]
    return t, d


def mcq_top1(model, kind, tok, name, fy, rev_b, mults, hf):
    enc = (lambda s: tok.encode(s, add_special_tokens=False)) if hf else tok.encode
    pre = (tok.encode(_prompt(name, fy), add_special_tokens=True) if hf else tok.encode(_prompt(name, fy)))
    t, dist = candidates(rev_b, mults)
    nt = H.value_recall_nll(model, kind, tok, pre, enc(f"{t} billion"))
    nds = [H.value_recall_nll(model, kind, tok, pre, enc(f"{d} billion")) for d in dist]
    nds = [x for x in nds if np.isfinite(x)]
    if not nds or not np.isfinite(nt):
        return None, None
    return int(all(nt < x for x in nds)), 1 + len(nds)


def _prompt(name, fy):
    return f"{name} reported total annual revenue for fiscal year {fy} of approximately $"


def gen_value(model, kind, tok, name, fy, hf):
    pre = (tok.encode(_prompt(name, fy), add_special_tokens=True) if hf else tok.encode(_prompt(name, fy)))
    txt = H.generate_text(model, kind, tok, pre, n_new=10)
    m = re.search(r"(\d+(?:[.,]\d+)?)", txt.replace(",", ""))
    return float(m.group(1)) if m else None


def run(name, loader, facts, hf, cut):
    out = loader()
    model, tok = (out[0], out[1]) if hf else (out[0], H.gpt2_tokenizer())
    kind = "hf-causal" if hf else "chrono-gpt"
    rows = []
    for _, f in facts.iterrows():
        w, nc = mcq_top1(model, kind, tok, f["name"], int(f["fy"]), float(f["rev_b"]), WIDE, hf)
        tt, _ = mcq_top1(model, kind, tok, f["name"], int(f["fy"]), float(f["rev_b"]), TIGHT, hf)
        g = gen_value(model, kind, tok, f["name"], int(f["fy"]), hf)
        rev = float(f["rev_b"])
        gerr = abs(g - rev) / rev if (g is not None and rev > 0) else np.nan
        rows.append({"model": name, "cutoff": cut, "tic": f["tic"], "fy": int(f["fy"]), "rev_b": rev,
                     "wide_top1": w, "n_cand": nc, "tight_top1": tt, "gen_err": gerr})
    import torch; del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(f"  {name}: wide={df.wide_top1.mean():.2f} tight={df.tight_top1.mean():.2f} "
          f"gen<=10%={np.mean(df.gen_err<=0.10):.2f}", flush=True)
    return df


def main():
    con = duckdb.connect(str(Path(__file__).resolve().parents[2] / "data" / "barj_master.duckdb"), read_only=True)
    facts = load_facts(con); con.close()
    print(f"[facts] {len(facts)} famous-firm facts", flush=True)
    parts = []
    for repo in discover_chrono():
        cut = int(repo.rsplit("-", 1)[-1][:4])
        parts.append(run(f"chrono-{cut}", lambda r=repo: H.load_chrono_gpt(r), facts, False, cut))
    from vr_02_value_recall import FRONTIER_CUTOFF
    for fp in discover_frontier():
        nm = os.path.basename(fp)
        try:
            parts.append(run(nm, lambda p=fp: H.load_hf_causal(p), facts, True, FRONTIER_CUTOFF.get(nm, 2024)))
        except Exception as e:  # noqa: BLE001
            print(f"  [{nm} FAILED: {str(e).splitlines()[0][:80]}]", flush=True)

    res = pd.concat([p for p in parts if len(p)], ignore_index=True)
    res.to_parquet(OUT / "specificity.parquet", index=False)
    summ = {"wide_distractors": WIDE, "tight_distractors": TIGHT, "n_facts": int(len(facts)), "by_model": {}}
    for m, d in res.groupby("model"):
        summ["by_model"][m] = {
            "wide_top1": round(float(d.wide_top1.mean()), 3),
            "tight_top1": round(float(d.tight_top1.mean()), 3),
            "wide_minus_tight": round(float(d.wide_top1.mean() - d.tight_top1.mean()), 3),
            "gen_within_5pct": round(float(np.mean(d.gen_err <= 0.05)), 3),
            "gen_within_10pct": round(float(np.mean(d.gen_err <= 0.10)), 3),
            "gen_within_20pct": round(float(np.mean(d.gen_err <= 0.20)), 3),
            "mcq_minus_gen10": round(float(d.wide_top1.mean() - np.mean(d.gen_err <= 0.10)), 3),
            "chance_wide": round(float((1.0 / d.n_cand).mean()), 3),
            "is_baseline": m.startswith("chrono"), "n": int(len(d))}
    (OUT / "specificity.json").write_text(json.dumps(summ, indent=2))
    print("\n=== specificity (memorisation vs estimation) ===")
    for m, v in sorted(summ["by_model"].items(), key=lambda kv: -kv[1]["tight_top1"]):
        print(f"  {m:20s} wide={v['wide_top1']:.2f} tight={v['tight_top1']:.2f} "
              f"(Δ{v['wide_minus_tight']:+.2f})  gen<=10%={v['gen_within_10pct']:.2f}")
    print(f"[written] {OUT/'specificity.json'}")


if __name__ == "__main__":
    main()
