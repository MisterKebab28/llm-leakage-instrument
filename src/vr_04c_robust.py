#!/usr/bin/env python
"""
vr_04c_robust.py -- robustness of the recall->trading transmission correlation (CPU).

Reviewer critique (correct): the +0.62 transmission corr is underpowered (N~8, heterogeneous
architectures, no CI, no leave-one-out, no rank correlation, an admitted outlier). This re-analysis
addresses all of it on whatever model set is present in BOTH transmission.parquet (directional) and
value_recall_cf.json (capacity):
  - Pearson AND Spearman corr(capacity, realized leakage = AUC_seen - 0.5)
  - bootstrap 95% CI over models
  - leave-one-model-out range (flags the most influential model, e.g. Llama)

Output: outputs/leakage/transmission_robust.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vr_04b_transmission_auc import auc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"


def main():
    df = pd.read_parquet(OUT / "transmission.parquet")
    df = df.assign(y=(df["true_dir"] == "higher").astype(int))
    cap = {m: v["share_true_top1"] for m, v in json.load(open(OUT / "value_recall_cf.json"))["by_model"].items()}

    pts = []
    for m, d in df.groupby("model"):
        if m.startswith("chrono") or m not in cap:
            continue
        s = d[d.seen == 1]
        a = auc(s["lean_higher_margin"].to_numpy(), s["y"].to_numpy()) if len(s) else float("nan")
        if np.isfinite(a):
            pts.append((m, cap[m], a - 0.5))      # (model, capacity, realized leakage)
    pts.sort(key=lambda x: -x[1])
    models = [p[0] for p in pts]
    x = np.array([p[1] for p in pts]); yv = np.array([p[2] for p in pts])
    n = len(pts)

    pear = float(np.corrcoef(x, yv)[0, 1]) if n >= 3 and x.std() > 0 and yv.std() > 0 else float("nan")
    spear = float(stats.spearmanr(x, yv).statistic) if n >= 3 else float("nan")
    # bootstrap CI over models
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], yv[idx]
        if xi.std() > 0 and yi.std() > 0:
            bs.append(np.corrcoef(xi, yi)[0, 1])
    ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if bs else [None, None]
    # leave-one-out
    loo = {}
    for i in range(n):
        xi = np.delete(x, i); yi = np.delete(yv, i)
        loo[models[i]] = round(float(np.corrcoef(xi, yi)[0, 1]), 3) if xi.std() > 0 and yi.std() > 0 else None
    loo_vals = [v for v in loo.values() if v is not None]

    res = {
        "n_models": n, "models": models,
        "pearson": round(pear, 3), "spearman": round(spear, 3),
        "bootstrap_ci95": ci,
        "leave_one_out_pearson": loo,
        "loo_range": [min(loo_vals), max(loo_vals)] if loo_vals else None,
        "most_influential": (min(loo, key=lambda k: loo[k]) if loo_vals else None),
        "verdict": ("Report the transmission link with these robustness stats. If the bootstrap CI "
                    "spans ~0 or LOO swings sign, downgrade from 'result' to 'suggestive, "
                    "underpowered' and do not headline the point estimate."),
        "per_model": {m: {"capacity": round(c, 3), "realized_leakage": round(r, 3)} for m, c, r in pts},
    }
    (OUT / "transmission_robust.json").write_text(json.dumps(res, indent=2))
    print(f"N={n}  Pearson={res['pearson']}  Spearman={res['spearman']}  bootCI={ci}")
    print(f"LOO range={res['loo_range']}  most influential (drop->lowest corr)={res['most_influential']}")
    print(f"[written] {OUT/'transmission_robust.json'}")


if __name__ == "__main__":
    main()
