#!/usr/bin/env python
"""
vr_10b_panel.py -- the decisive confound check on agentic transmission (CPU).

The year-FE result (recall_hit -> accuracy, t=4.07 for Qwen2.5-7B-Instruct) could be a PROMINENCE
confound: recall-hit cells are famous mega-caps that may be more predictable for reasons other than
memory. The clean test adds FIRM fixed effects -> identifies WITHIN a firm, across years: does
recalling THAT year's value predict getting THAT year's direction right?

For each outputs/leakage/agentic_cells_<model>.parquet:
  - year-FE-only diff (reproduces vr_10)
  - TWO-WAY (firm+year) FE slope of `correct` on `recall_hit`, via iterative demeaning,
    with FIRM-CLUSTERED standard errors -> t, p.
Survives firm FE => year-specific memory transmits (real). Vanishes => it was prominence.

Output: outputs/leakage/agentic_panel_fe.json
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"


def demean(s: pd.Series, key: pd.Series) -> pd.Series:
    return s - s.groupby(key).transform("mean")


def two_way_fe(d: pd.DataFrame):
    """Slope of correct~recall_hit absorbing firm+year FE (iterative demeaning) + firm-clustered SE."""
    y = d["correct"].astype(float).reset_index(drop=True)
    x = d["recall_hit"].astype(float).reset_index(drop=True)
    tic = d["tic"].reset_index(drop=True); fy = d["fy"].reset_index(drop=True)
    for _ in range(30):
        y = demean(demean(y, tic), fy)
        x = demean(demean(x, tic), fy)
    sxx = float((x * x).sum())
    if sxx < 1e-9:
        return None  # no within-firm-year variation in recall (e.g. degenerate)
    beta = float((x * y).sum() / sxx)
    e = y - beta * x
    # firm-clustered SE
    g = (x * e)
    meat = float(sum(g.groupby(tic).sum() ** 2))
    se = float(np.sqrt(meat) / sxx)
    t = beta / se if se > 0 else float("nan")
    # dof ~ n_clusters-1
    nclu = tic.nunique()
    p = float(2 * stats.t.sf(abs(t), df=max(1, nclu - 1)))
    return {"beta_twowayFE": round(beta, 4), "cluster_se": round(se, 4), "t": round(t, 2),
            "p": round(p, 4), "n_firms": int(nclu), "n": int(len(d))}


def main():
    res = {}
    for fp in sorted(glob.glob(str(OUT / "agentic_cells_*.parquet"))):
        name = os.path.basename(fp)[len("agentic_cells_"):-len(".parquet")]
        d = pd.read_parquet(fp)
        # year-FE-only diff (reproduce vr_10)
        d = d.assign(correct_fe=d["correct"] - d.groupby("fy")["correct"].transform("mean"))
        g1, g0 = d[d.recall_hit == 1], d[d.recall_hit == 0]
        diff_year = float(g1["correct_fe"].mean() - g0["correct_fe"].mean()) if len(g1) and len(g0) else None
        tw = two_way_fe(d)
        res[name] = {
            "frac_higher": round(float((d.decision == "higher").mean()), 3) if "decision" in d else None,
            "recall_rate": round(float(d.recall_hit.mean()), 3),
            "diff_yearFE_only": round(diff_year, 4) if diff_year is not None else None,
            "twoway_firm_year_FE": tw,
            "verdict": ("survives firm FE" if tw and tw["p"] < 0.05 and tw["beta_twowayFE"] > 0
                        else "vanishes under firm FE (prominence confound)" if tw
                        else "no within-firm variation (degenerate decisions)"),
        }
        v = res[name]; t = v["twoway_firm_year_FE"]
        print(f"  {name}: yearFE_diff={v['diff_yearFE_only']}  ->  twoway "
              f"{'beta='+str(t['beta_twowayFE'])+' t='+str(t['t'])+' p='+str(t['p']) if t else 'n/a'}  "
              f"[{v['verdict']}]")
    (OUT / "agentic_panel_fe.json").write_text(json.dumps(res, indent=2))
    print(f"[written] {OUT/'agentic_panel_fe.json'}")


if __name__ == "__main__":
    main()
