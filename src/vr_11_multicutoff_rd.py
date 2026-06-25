#!/usr/bin/env python
"""
vr_11_multicutoff_rd.py -- staggered multi-cutoff regression discontinuity (the causal core).

Models have different training cutoffs. For each (model, firm, year) cell define
rel = fy - cutoff  (rel<0 = fact was in training/memorisable; rel>=0 = after cutoff). Pooling across
models with different cutoffs gives a sharp event-time axis centred on each model's cutoff. We measure
two outcomes per cell and look for a DISCONTINUITY at rel=0:

  recall_hit : tight-distractor (+/-15%, magnitude-robust) value recall  -> CAPACITY
  correct    : model's directional bet matches realised split-adj return  -> TRADING

Predictions:
  - recall: SHARP drop at rel=0 (memorisation is cutoff-bounded; competent estimation would be smooth)
    => a recall discontinuity is the causal signature of memorisation, not reasoning.
  - trading: NO discontinuity if memorisation does not transmit (Branch B) -> causal confirmation.

Run under conda `base` (GPU). Output: outputs/leakage/rd_cells.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402
from vr_02_value_recall import FRONTIER_CUTOFF, discover_frontier  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
YEARS = list(range(2013, 2026))          # 2013-2025: gives post-cutoff facts for 2023-24-cutoff models
N_FIRMS = 150
TIGHT = [0.85, 0.93, 1.08, 1.18]         # magnitude-robust = precise-memory flag


def clean(conm):
    n = conm.title()
    for s in (" Inc", " Corp", " Co", " Plc", " Ltd", " Group", " Holdings", " S A", " Cl A", " -Cl A"):
        if n.endswith(s):
            n = n[:-len(s)]
    return n.strip(" .-&")


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def panel(con, n_firms=N_FIRMS):
    tics = [r[0] for r in con.execute(
        f"""SELECT tic FROM w_comp_na_daily_all__funda
            WHERE datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
              AND fyear=2022 AND sale>500 AND tic IS NOT NULL
            ORDER BY sale DESC LIMIT {n_firms}""").fetchall()]
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
            FROM p GROUP BY 1,2 HAVING count(*)>=150""").fetchdf()
    px["true_dir"] = np.where(px["c1"] / px["c0"] - 1.0 >= 0, "higher", "lower")
    df = rev.merge(px[["tic", "fy", "true_dir"]], on=["tic", "fy"], how="inner")
    df["name"] = df["conm"].map(clean)
    return df


def main():
    con = duckdb.connect(str(DB), read_only=True)
    cells = panel(con)
    con.close()
    print(f"[panel] {len(cells)} cells, {cells.tic.nunique()} firms, years {YEARS[0]}-{YEARS[-1]}, "
          f"post-2022 facts={int((cells.fy>2022).sum())}", flush=True)
    allrows = []
    for fp in discover_frontier():
        name = os.path.basename(fp)
        cutoff = FRONTIER_CUTOFF.get(name, 2024)
        try:
            model, tok, _ = H.load_hf_causal(fp)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name} load failed: {str(e).splitlines()[0][:70]}]", flush=True); continue
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        rows = []
        for _, f in cells.iterrows():
            nm, fy, rev = f["name"], int(f["fy"]), float(f["rev_b"])
            pre = tok.encode(f"{nm} reported total annual revenue for fiscal year {fy} of approximately $",
                             add_special_tokens=True)
            t = fmt(rev); dist = [x for x in sorted({fmt(rev * m) for m in TIGHT}) if x != t and x >= 0.1]
            nt = H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{t} billion"))
            nds = [H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{d} billion")) for d in dist]
            nds = [x for x in nds if np.isfinite(x)]
            recall_hit = int(np.isfinite(nt) and nds and all(nt < x for x in nds))
            dp = (f"Over the {fy} calendar year, did shares of {nm} finish higher or lower than where "
                  f"they started? Answer with one word:")
            dpre = tok.encode(dp, add_special_tokens=True)
            nh = H.value_recall_nll(model, "hf-causal", tok, dpre, enc(" higher"))
            nl = H.value_recall_nll(model, "hf-causal", tok, dpre, enc(" lower"))
            if not (np.isfinite(nh) and np.isfinite(nl)):
                continue
            pred = "higher" if nh < nl else "lower"
            rows.append({"model": name, "cutoff": cutoff, "tic": f["tic"], "fy": fy,
                         "rel": fy - cutoff, "recall_hit": recall_hit,
                         "correct": int(pred == f["true_dir"]), "true_dir": f["true_dir"]})
        import torch; del model; torch.cuda.empty_cache()
        d = pd.DataFrame(rows)
        pre_r = d[d.rel < 0].recall_hit.mean() if (d.rel < 0).any() else float("nan")
        post_r = d[d.rel >= 0].recall_hit.mean() if (d.rel >= 0).any() else float("nan")
        print(f"  {name} (cut {cutoff}): recall pre={pre_r:.2f} post={post_r:.2f}  "
              f"n_pre={int((d.rel<0).sum())} n_post={int((d.rel>=0).sum())}", flush=True)
        allrows.append(d)
    res = pd.concat([r for r in allrows if len(r)], ignore_index=True)
    res.to_parquet(OUT / "rd_cells.parquet", index=False)
    print(f"\n[written] {OUT/'rd_cells.parquet'}  ({len(res)} cells, {res.model.nunique()} models)")
    print(f"  pooled recall: pre-cutoff {res[res.rel<0].recall_hit.mean():.3f} vs "
          f"post {res[res.rel>=0].recall_hit.mean():.3f}")
    print(f"  pooled trading acc: pre {res[res.rel<0].correct.mean():.3f} vs "
          f"post {res[res.rel>=0].correct.mean():.3f}")


if __name__ == "__main__":
    main()
