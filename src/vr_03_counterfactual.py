#!/usr/bin/env python
"""
vr_03_counterfactual.py -- value-recall as WITHIN-MODEL counterfactual discrimination.

vr_02 showed the cross-cutoff RD is null on chrono-gpt (1.5B lacks the capacity to
memorise specific revenue figures) and that raw cross-tokenizer NLL can't compare
chrono vs Qwen3. The correct capacity instrument for a single (esp. frontier) model
is counterfactual discrimination, which is tokenizer-internal and needs no staggered
cutoffs:

  For fact (company C, fiscal year Y, true revenue V_true), score the SAME sentence
  with the true value and with plausible distractors {0.5,0.7,1.4,2.0}xV_true.
  recall_margin = mean(NLL_distractor) - NLL_true.   margin > 0  <=>  the model
  finds the REALIZED value more likely than counterfactuals = value-recall capacity.

This is the headline capacity probe (v5 section 2). Comparisons are within-model
(same tokenizer), so chrono-gpt (no-leak baseline, expect ~0) and Qwen3-8B (frontier,
expect > 0) are each internally valid; the chrono cross-cutoff split additionally tests
whether any discrimination appears only once the cutoff covers the fact year.

Run under conda `base` (GPU). Output: outputs/leakage/value_recall_cf.{parquet,json}
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402
from vr_02_value_recall import (NAMES, BASKET, FY_LO, FY_HI, load_facts,  # noqa: E402
                                discover_chrono, discover_frontier, FRONTIER_CUTOFF)

OUT = ROOT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"
DB = Path(__file__).resolve().parents[2] / "data" / "barj_master.duckdb"

MULTS = [0.5, 0.7, 1.4, 2.0]


def candidates(true_b: int) -> tuple[int, list[int]]:
    dist = sorted({int(round(true_b * m)) for m in MULTS} - {true_b})
    dist = [d for d in dist if d >= 1]
    return true_b, dist


def prefix_for(name: str, fy: int) -> str:
    return f"{name} reported total annual revenue for fiscal year {fy} of approximately $"


def score_fact(model, kind, tok, name, fy, true_b, hf=False):
    pre = tok.encode(prefix_for(name, fy), add_special_tokens=True) if hf \
        else tok.encode(prefix_for(name, fy))
    enc = (lambda s: tok.encode(s, add_special_tokens=False)) if hf else tok.encode
    t, dist = candidates(true_b)
    nll_t = H.value_recall_nll(model, kind, tok, pre, enc(f"{t} billion"))
    nll_d = [H.value_recall_nll(model, kind, tok, pre, enc(f"{d} billion")) for d in dist]
    nll_d = [x for x in nll_d if np.isfinite(x)]
    if not nll_d or not np.isfinite(nll_t):
        return None
    rank = 1 + sum(x < nll_t for x in nll_d)        # 1 = true is most likely
    return {"nll_true": nll_t, "nll_dist_mean": float(np.mean(nll_d)),
            "margin": float(np.mean(nll_d) - nll_t), "rank_true": rank,
            "n_cand": 1 + len(nll_d), "top1": int(rank == 1)}


def run_model(name, loader, facts, hf=False) -> pd.DataFrame:
    out = loader()
    model, tok = (out[0], out[1]) if hf else (out[0], H.gpt2_tokenizer())
    meta_kind = "hf-causal" if hf else "chrono-gpt"
    rows = []
    for _, f in facts.iterrows():
        r = score_fact(model, meta_kind, tok, f["name"], f["fy"], int(f["rev_b"]), hf=hf)
        if r:
            rows.append({"model": name, "tic": f["tic"], "fy": int(f["fy"]),
                         "rev_b": int(f["rev_b"]), **r})
    import torch; del model; torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    facts = load_facts(con)
    con.close()
    print(f"[facts] {len(facts)} facts, {facts['tic'].nunique()} firms")

    parts = []
    for repo in discover_chrono():
        cut = int(repo.rsplit("-", 1)[-1][:4])
        df = run_model(f"chrono-{cut}", lambda r=repo: H.load_chrono_gpt(r), facts, hf=False)
        df["cutoff"] = cut
        parts.append(df)
        print(f"  scored chrono-{cut}: mean margin={df['margin'].mean():+.3f} "
              f"top1={df['top1'].mean():.2f}")

    for fpath in discover_frontier():
        name = os.path.basename(fpath)
        try:
            df = run_model(name, lambda p=fpath: H.load_hf_causal(p), facts, hf=True)
            df["cutoff"] = FRONTIER_CUTOFF.get(name, 2024)
            parts.append(df)
            print(f"  scored {name}: mean margin={df['margin'].mean():+.3f} "
                  f"top1={df['top1'].mean():.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{name} failed: {str(e).splitlines()[0][:100]}]")

    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(OUT / "value_recall_cf.parquet", index=False)

    summary = {"probe": "within-model counterfactual value-recall (revenue)",
               "distractor_multipliers": MULTS, "n_facts": int(len(facts)),
               "by_model": {}}
    for m, d in res.groupby("model"):
        chance_top1 = float((1.0 / d["n_cand"]).mean())
        summary["by_model"][m] = {
            "mean_margin_nll": round(float(d["margin"].mean()), 4),
            "median_margin": round(float(d["margin"].median()), 4),
            "share_true_top1": round(float(d["top1"].mean()), 3),
            "chance_top1": round(chance_top1, 3),
            "mean_rank_true": round(float(d["rank_true"].mean()), 3),
            "n": int(len(d)),
        }
    # chrono cross-cutoff: does discrimination appear only once cutoff covers the year?
    ch = res[res.model.str.startswith("chrono")].copy()
    ch["seen"] = ch["cutoff"] > ch["fy"]   # FY-YYYY revenue reported early YYYY+1 (PIT)
    summary["chrono_seen_vs_unseen"] = {
        "margin_seen": round(float(ch[ch.seen]["margin"].mean()), 4),
        "margin_unseen": round(float(ch[~ch.seen]["margin"].mean()), 4),
        "top1_seen": round(float(ch[ch.seen]["top1"].mean()), 3),
        "top1_unseen": round(float(ch[~ch.seen]["top1"].mean()), 3),
    }
    summary["interpretation"] = (
        "margin>0 and share_true_top1>chance => the model prefers the REALIZED value over "
        "counterfactuals = value-recall capacity. Expectation: chrono-gpt ~chance (1.5B "
        "no-capacity floor); Qwen3-8B above chance if frontier models carry the capacity "
        "that LLM-trading studies would mistake for skill.")
    (OUT / "value_recall_cf.json").write_text(json.dumps(summary, indent=2))

    print("\n=== counterfactual value-recall by model ===")
    for m, v in summary["by_model"].items():
        print(f"  {m:16s} margin={v['mean_margin_nll']:+.3f}  top1={v['share_true_top1']:.2f} "
              f"(chance {v['chance_top1']:.2f})  mean_rank={v['mean_rank_true']:.2f}")
    print("chrono seen vs unseen:", summary["chrono_seen_vs_unseen"])
    print(f"\n[written] {OUT/'value_recall_cf.json'}")


if __name__ == "__main__":
    main()
