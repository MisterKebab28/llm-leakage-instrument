#!/usr/bin/env python
"""
vr_09_decomposition.py -- WITHIN-MODEL, per-decision transmission (the powered fix + point c).

The cross-model transmission corr (N~7 models, CI [-0.86,+0.89]) is underpowered. This re-casts
the estimand WITHIN a single model at the level of (firm, year) DECISIONS, so N = thousands:

  For each (firm, year) cell:
    recall_hit  = does the instrument detect the model recalls the firm's revenue?  (capacity)
    correct     = is the model's directional bet (stock up/down that year) right?     (decision)
  Transmission = is `correct` higher where `recall_hit=1` than where `=0`, NET OF YEAR fixed
  effects (absorbs the stocks-mostly-rise base rate)? This decomposes the model's directional
  "skill" into a recall-detectable (leakage) component vs the rest -- the leaderboard indictment
  turned into a measurement. recall (revenue) and the decision (return direction) are distinct
  facts, so the link is non-circular.

Run under conda `base` (GPU). Usage: vr_09_decomposition.py [model_dir ...]  (default: 3 strong)
Output: outputs/leakage/decomposition.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
MULTS = [0.5, 0.7, 1.4, 2.0]
YEARS = list(range(2016, 2024))
N_FIRMS = 300
DEFAULT = [r"E:\models\Qwen2.5-14B", r"E:\models\Mistral-Nemo-Base-2407", r"E:\models\Llama-3.1-8B"]


def clean(conm):
    n = conm.title()
    for s in (" Inc", " Corp", " Co", " Plc", " Ltd", " Group", " Holdings", " S A", " Cl A", " -Cl A"):
        if n.endswith(s):
            n = n[:-len(s)]
    return n.strip(" .-&")


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def panel(con):
    # revenue facts (firm, year) for a broad liquid-ish set, joined to annual price direction
    tics = [r[0] for r in con.execute(
        f"""SELECT tic FROM w_comp_na_daily_all__funda
            WHERE datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
              AND fyear=2022 AND sale>3000 AND tic IS NOT NULL
            ORDER BY sale DESC LIMIT {N_FIRMS}""").fetchall()]
    tl = ",".join(f"'{t}'" for t in tics)
    rev = con.execute(
        f"""SELECT tic, CAST(fyear AS INT) fy, conm, round(sale/1000.0,1) rev_b
            FROM w_comp_na_daily_all__funda
            WHERE tic IN ({tl}) AND datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
              AND fyear BETWEEN {YEARS[0]} AND {YEARS[-1]} AND sale>0
            QUALIFY row_number() OVER (PARTITION BY tic,fyear ORDER BY sale DESC)=1""").fetchdf()
    con.execute("SET TimeZone='UTC'")
    px = con.execute(
        f"""WITH p AS (SELECT raw_symbol, CAST(timestamp AS DATE) d, close, year(timestamp) yr
                       FROM polygon_ohlcv_1d WHERE raw_symbol IN ({tl}) AND close>0 AND volume>0
                         AND year(timestamp) BETWEEN {YEARS[0]} AND {YEARS[-1]}
                         AND NOT (month(timestamp)=1 AND day(timestamp)=1))
            SELECT raw_symbol tic, yr fy, arg_min(close,d) c0, arg_max(close,d) c1, count(*) n
            FROM p GROUP BY 1,2 HAVING count(*)>=200""").fetchdf()
    px["true_dir"] = np.where(px["c1"] / px["c0"] - 1.0 >= 0, "higher", "lower")
    df = rev.merge(px[["tic", "fy", "true_dir"]], on=["tic", "fy"], how="inner")
    df["name"] = df["conm"].map(clean)
    return df


def main():
    models = sys.argv[1:] or DEFAULT
    con = duckdb.connect(str(DB), read_only=True)
    cells = panel(con)
    con.close()
    print(f"[panel] {len(cells)} (firm,year) cells, {cells['tic'].nunique()} firms, "
          f"base-rate higher={ (cells.true_dir=='higher').mean():.2f}", flush=True)

    results = {}
    for mdir in models:
        name = os.path.basename(mdir.rstrip("/\\"))
        try:
            model, tok, _ = H.load_hf_causal(mdir)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name} load failed: {str(e).splitlines()[0][:70]}]", flush=True); continue
        rows = []
        for _, f in cells.iterrows():
            nm, fy, rev = f["name"], int(f["fy"]), float(f["rev_b"])
            rp = f"{nm} reported total annual revenue for fiscal year {fy} of approximately $"
            pre = tok.encode(rp, add_special_tokens=True)
            enc = lambda s: tok.encode(s, add_special_tokens=False)
            t = fmt(rev); dist = [x for x in sorted({fmt(rev*m) for m in MULTS}) if x != t and x >= 0.1]
            nt = H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{t} billion"))
            nds = [H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{d} billion")) for d in dist]
            nds = [x for x in nds if np.isfinite(x)]
            recall_hit = int(np.isfinite(nt) and nds and all(nt < x for x in nds))
            # directional decision
            dp = (f"Over the {fy} calendar year, did shares of {nm} finish higher or lower than "
                  f"where they started? Answer with one word:")
            dpre = tok.encode(dp, add_special_tokens=True)
            nh = H.value_recall_nll(model, "hf-causal", tok, dpre, enc(" higher"))
            nl = H.value_recall_nll(model, "hf-causal", tok, dpre, enc(" lower"))
            if not (np.isfinite(nh) and np.isfinite(nl)):
                continue
            pred = "higher" if nh < nl else "lower"
            rows.append({"tic": f["tic"], "fy": fy, "recall_hit": recall_hit,
                         "correct": int(pred == f["true_dir"]), "true_dir": f["true_dir"]})
        import torch; del model; torch.cuda.empty_cache()
        d = pd.DataFrame(rows)
        # year-fixed-effect adjusted: demean `correct` within year, compare by recall_hit
        d["correct_fe"] = d["correct"] - d.groupby("fy")["correct"].transform("mean")
        g1 = d[d.recall_hit == 1]; g0 = d[d.recall_hit == 0]
        # difference in FE-adjusted accuracy + a 2-sample t on the demeaned outcome
        from scipy import stats
        diff_fe = float(g1["correct_fe"].mean() - g0["correct_fe"].mean())
        tt = stats.ttest_ind(g1["correct_fe"], g0["correct_fe"], equal_var=False)
        # bootstrap CI on the raw acc difference
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
            a1 = s[s.recall_hit == 1]["correct"]; a0 = s[s.recall_hit == 0]["correct"]
            if len(a1) and len(a0):
                bs.append(a1.mean() - a0.mean())
        results[name] = {
            "n_cells": int(len(d)), "n_recall_hit": int(d.recall_hit.sum()),
            "recall_rate": round(float(d.recall_hit.mean()), 3),
            "acc_recall_hit": round(float(g1.correct.mean()), 3),
            "acc_recall_miss": round(float(g0.correct.mean()), 3),
            "acc_diff_raw": round(float(g1.correct.mean() - g0.correct.mean()), 3),
            "acc_diff_yearFE": round(diff_fe, 3),
            "t_stat": round(float(tt.statistic), 2), "p_value": round(float(tt.pvalue), 4),
            "boot_ci95_raw_diff": [round(float(np.percentile(bs, 2.5)), 3),
                                   round(float(np.percentile(bs, 97.5)), 3)] if bs else None,
            "down_year_acc_hit": round(float(g1[g1.true_dir == "lower"].correct.mean()), 3) if len(g1[g1.true_dir == "lower"]) else None,
            "down_year_acc_miss": round(float(g0[g0.true_dir == "lower"].correct.mean()), 3) if len(g0[g0.true_dir == "lower"]) else None,
        }
        r = results[name]
        print(f"  {name}: acc|hit={r['acc_recall_hit']} vs miss={r['acc_recall_miss']} "
              f"(FE diff {r['acc_diff_yearFE']:+.3f}, t={r['t_stat']}, p={r['p_value']}, "
              f"CI{r['boot_ci95_raw_diff']}, N={r['n_cells']})", flush=True)

    out = {"design": "within-model per-decision transmission; correct ~ recall_hit + year FE",
           "n_cells_panel": int(len(cells)), "years": YEARS, "by_model": results,
           "interpretation": ("acc_diff_yearFE > 0 with t large / CI excluding 0 => the model's "
                              "directional decisions are more accurate on firm-years it can recall = "
                              "recall capacity transmits to the decision, measured at high power "
                              "(N=cells). This decomposes backtest 'skill' into a leakage-detectable "
                              "component vs the rest.")}
    (OUT / "decomposition.json").write_text(json.dumps(out, indent=2))
    print(f"\n[written] {OUT/'decomposition.json'}")


if __name__ == "__main__":
    main()
