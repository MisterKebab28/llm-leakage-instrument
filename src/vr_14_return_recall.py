#!/usr/bin/env python
"""
vr_14_return_recall.py -- does the model memorise the CONTAMINATING VARIABLE (returns), not just revenue?

Construct-validity gap (referee, correct): the revenue-recall instrument measures recall of one
fundamental, but the contamination a backtest fears is memorised PRICES/RETURNS. This probes recall of
the realised annual return itself, two ways, per (firm, year):

  dir_recall : the model's recalled SIGN of the return ("shares finished higher/lower") vs the realised
               sign -> minimal contamination (knowing the sign is enough to trade).
  mag_recall : counterfactual over return MAGNITUDE (true rounded return vs sign/magnitude distractors)
               -> precise price memory.

Decomposed seen (pre-cutoff) vs unseen (post-cutoff) per model. If return-recall is above chance for
seen and collapses for unseen, the contamination capacity exists for the RELEVANT variable; if it is at
chance, the field's memorised-price fear is weaker than the (strong) memorised-fundamental capacity.

Run under conda `base` (GPU). Output: outputs/leakage/return_recall.{parquet,json}
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
from vr_02_value_recall import FRONTIER_CUTOFF, discover_frontier  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
YEARS = list(range(2013, 2026))
N_FIRMS = 120


def clean(conm):
    n = conm.title()
    for s in (" Inc", " Corp", " Co", " Plc", " Ltd", " Group", " Holdings", " S A", " Cl A", " -Cl A"):
        if n.endswith(s):
            n = n[:-len(s)]
    return n.strip(" .-&")


def panel(con):
    tics = [r[0] for r in con.execute(
        f"""SELECT tic FROM w_comp_na_daily_all__funda WHERE datafmt='STD' AND indfmt='INDL'
            AND popsrc='D' AND consol='C' AND fyear=2022 AND sale>3000 AND tic IS NOT NULL
            ORDER BY sale DESC LIMIT {N_FIRMS}""").fetchall()]
    tl = ",".join(f"'{t}'" for t in tics)
    nm = con.execute(f"""SELECT tic, any_value(conm) conm FROM w_comp_na_daily_all__funda
                         WHERE tic IN ({tl}) GROUP BY tic""").fetchdf()
    con.execute("SET TimeZone='UTC'")
    px = con.execute(
        f"""WITH p AS (SELECT raw_symbol, CAST(timestamp AS DATE) d, close, year(timestamp) yr
                       FROM polygon_ohlcv_1d WHERE raw_symbol IN ({tl}) AND close>0 AND volume>0
                         AND year(timestamp) BETWEEN {YEARS[0]} AND {YEARS[-1]}
                         AND NOT (month(timestamp)=1 AND day(timestamp)=1))
            SELECT raw_symbol tic, yr fy, arg_min(close,d) c0, arg_max(close,d) c1
            FROM p GROUP BY 1,2 HAVING count(*)>=150""").fetchdf()
    px["ret"] = (px["c1"] / px["c0"] - 1.0) * 100.0
    df = px.merge(nm, on="tic", how="left")
    df["name"] = df["conm"].map(clean)
    return df.dropna(subset=["name"])


def mag_candidates(ret):
    t = int(round(ret / 5.0) * 5)                      # round true to nearest 5%
    grid = {t, -t, t + 25, t - 25, 10, -10, 40, -20}   # sign-flip + magnitude + anchors
    return t, [x for x in sorted(grid) if x != t][:5]


def run(name, loader, cells, cut):
    model, tok, _ = loader()
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    rows = []
    for _, f in cells.iterrows():
        nm, fy, ret = f["name"], int(f["fy"]), float(f["ret"])
        # direction recall (sign memory)
        dp = tok.encode(f"Over the {fy} calendar year, shares of {nm} finished", add_special_tokens=True)
        nh = H.value_recall_nll(model, "hf-causal", tok, dp, enc(" higher"))
        nl = H.value_recall_nll(model, "hf-causal", tok, dp, enc(" lower"))
        dir_hit = int((nh < nl) == (ret >= 0)) if (np.isfinite(nh) and np.isfinite(nl)) else None
        # magnitude recall (counterfactual)
        mp = tok.encode(f"Over the {fy} calendar year, {nm} stock returned approximately ",
                        add_special_tokens=True)
        t, dist = mag_candidates(ret)
        def s(v):
            return f"{'+' if v >= 0 else ''}{v}%"
        nt = H.value_recall_nll(model, "hf-causal", tok, mp, enc(s(t)))
        nds = [x for x in (H.value_recall_nll(model, "hf-causal", tok, mp, enc(s(d))) for d in dist) if np.isfinite(x)]
        mag_hit = int(np.isfinite(nt) and nds and all(nt < x for x in nds)) if nds else None
        rows.append({"model": name, "tic": f["tic"], "fy": fy, "rel": fy - cut,
                     "seen": int(fy < cut), "dir_hit": dir_hit, "mag_hit": mag_hit,
                     "mag_nc": 1 + len(nds) if nds else None})
    import torch; del model; torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    se, un = d[d.seen == 1], d[d.seen == 0]
    print(f"  {name}: dir seen={se.dir_hit.mean():.2f} unseen={un.dir_hit.mean():.2f} | "
          f"mag seen={se.mag_hit.mean():.2f} unseen={un.mag_hit.mean():.2f} "
          f"(chance~{1/d.mag_nc.dropna().mean():.2f})", flush=True)
    return d


def main():
    con = duckdb.connect(str(DB), read_only=True)
    cells = panel(con); con.close()
    print(f"[panel] {len(cells)} (firm,year) cells, {cells.tic.nunique()} firms, "
          f"up-rate={(cells.ret>=0).mean():.2f}", flush=True)
    parts = []
    for fp in discover_frontier():
        nm = os.path.basename(fp)
        cut = FRONTIER_CUTOFF.get(nm, 2024)
        try:
            parts.append(run(nm, lambda p=fp: H.load_hf_causal(p), cells, cut))
        except Exception as e:  # noqa: BLE001
            print(f"  [{nm} failed: {str(e).splitlines()[0][:70]}]", flush=True)
    res = pd.concat([p for p in parts if len(p)], ignore_index=True)
    res.to_parquet(OUT / "return_recall.parquet", index=False)
    se, un = res[res.seen == 1], res[res.seen == 0]
    summ = {"n_cells": int(len(res)), "n_models": int(res.model.nunique()),
            "dir_recall_seen": round(float(se.dir_hit.mean()), 3), "dir_recall_unseen": round(float(un.dir_hit.mean()), 3),
            "mag_recall_seen": round(float(se.mag_hit.mean()), 3), "mag_recall_unseen": round(float(un.mag_hit.mean()), 3),
            "mag_chance": round(float(1 / res.mag_nc.dropna().mean()), 3),
            "interpretation": ("dir/mag recall >chance for seen and ->chance for unseen => the model "
                               "memorises the CONTAMINATING variable (returns), cutoff-bounded; at chance "
                               "=> memorised-price fear weaker than memorised-fundamental capacity.")}
    (OUT / "return_recall.json").write_text(json.dumps(summ, indent=2))
    print(f"\n=== return-recall: dir seen {summ['dir_recall_seen']} vs unseen {summ['dir_recall_unseen']} "
          f"(chance .50) | mag seen {summ['mag_recall_seen']} vs unseen {summ['mag_recall_unseen']} "
          f"(chance {summ['mag_chance']}) ===")
    print(f"[written] {OUT/'return_recall.json'}")


if __name__ == "__main__":
    main()
