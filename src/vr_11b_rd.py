#!/usr/bin/env python
"""
vr_11b_rd.py -- estimate the discontinuity at the cutoff (rel=0) for recall and trading (CPU).

Local-linear RD: y ~ 1 + post + rel + rel*post over a bandwidth |rel|<=h, jump = coef on `post`,
with MODEL-clustered standard errors. Run for recall_hit (expect a large negative jump = memorisation
is cutoff-bounded) and correct (expect ~0 = no transmission). Writes a figure + json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"
FIG = Path(__file__).resolve().parents[2] / "research_paper_leakage" / "figs"


def rd(d, ycol, h):
    s = d[(d.rel >= -h) & (d.rel <= h)].copy()
    post = (s.rel >= 0).astype(float).values
    rel = s.rel.astype(float).values
    X = np.column_stack([np.ones(len(s)), post, rel, rel * post])
    y = s[ycol].astype(float).values
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    # cluster by model
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in s.groupby("model").groups.items():
        Xg = X[[s.index.get_loc(i) for i in idx]]
        eg = e[[s.index.get_loc(i) for i in idx]]
        sc = Xg.T @ eg
        meat += np.outer(sc, sc)
    V = XtX_inv @ meat @ XtX_inv
    jump = float(beta[1]); se = float(np.sqrt(V[1, 1]))
    nclu = s.model.nunique()
    t = jump / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=max(1, nclu - 1)))
    return {"jump": round(jump, 4), "se": round(se, 4), "t": round(t, 2), "p": round(p, 4),
            "bandwidth": h, "n": int(len(s)), "n_models": int(nclu)}


def main():
    d = pd.read_parquet(OUT / "rd_cells.parquet").reset_index(drop=True)
    res = {"n_cells": int(len(d)), "n_models": int(d.model.nunique()),
           "pre_post_means": {
               "recall_pre": round(float(d[d.rel < 0].recall_hit.mean()), 3),
               "recall_post": round(float(d[d.rel >= 0].recall_hit.mean()), 3),
               "trading_pre": round(float(d[d.rel < 0].correct.mean()), 3),
               "trading_post": round(float(d[d.rel >= 0].correct.mean()), 3)}}
    for h in (4, 6):
        res[f"recall_RD_h{h}"] = rd(d, "recall_hit", h)
        res[f"trading_RD_h{h}"] = rd(d, "correct", h)
    (OUT / "rd_estimates.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if "RD" in k:
            print(f"  {k}: jump={v['jump']} (t={v['t']}, p={v['p']}, n={v['n']}, models={v['n_models']})")

    # figure: binned means vs rel
    g = d.groupby("rel").agg(recall=("recall_hit", "mean"), acc=("correct", "mean"),
                             n=("recall_hit", "size")).reset_index()
    g = g[g.n >= 20]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvline(0, color="k", ls="--", lw=1, label="training cutoff")
    ax.plot(g.rel, g.recall, "o-", color="#2c6fbb", label="value recall (capacity)")
    ax.plot(g.rel, g.acc, "s-", color="#e67e22", label="trading accuracy")
    ax.axhline(0.20, color="#2c6fbb", ls=":", lw=0.8)
    ax.set_xlabel("fiscal year relative to model training cutoff (rel = fy − cutoff)")
    ax.set_ylabel("rate"); ax.legend(fontsize=8)
    ax.set_title("Staggered multi-cutoff RD: recall drops at the cutoff, trading accuracy does not")
    fig.tight_layout(); fig.savefig(FIG / "fig5_multicutoff_rd.png", dpi=150); plt.close(fig)
    print(f"[written] {OUT/'rd_estimates.json'} + fig5_multicutoff_rd.png")


if __name__ == "__main__":
    main()
