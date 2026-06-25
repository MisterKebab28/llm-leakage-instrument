#!/usr/bin/env python
"""
vr_13_agentic_backtest.py -- realistic agentic portfolio backtest, decomposed by leakage (the
gold-standard transmission test; the StockBench/LiveTradeBench analog with a cutoff decomposition).

Each month, for a universe of liquid large-caps, the model receives a PIT context (trailing returns +
latest fundamentals, only data available as of the decision date) and makes a committed call per name:
OUTPERFORM or UNDERPERFORM the universe next month. We form an equal-weight long(OUTPERFORM)-short
(UNDERPERFORM) portfolio, realise next-month returns, and decompose:

  - pre-cutoff months (memorisation available) vs post-cutoff (FY2025+, none): is Sharpe higher pre?
  - recall-detectable name-months (instrument flags the name's revenue recalled) vs not.

If memorisation realises as trading skill -> pre-cutoff Sharpe > post AND performance concentrates on
recall-detectable picks. Null on both -> the capacity-without-realisation dissociation holds in the
realistic agentic setting the field actually scores.

Works with OPEN models (chat template, local GPU) and CLOSED APIs (reuses vr_C_api call fns; needs keys
+ budget). Run under conda `base`. Output: outputs/leakage/backtest_<model>.{parquet,json}
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
N_UNIV = 30
START, END = "2018-01-01", "2025-12-31"
CUTOFF_MONTH = "2024-01"          # months >= this are post-cutoff for the ~2023-cutoff roster
MULTS = [0.5, 0.7, 1.4, 2.0]


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def load_data(con):
    # universe: most liquid large-caps with full history
    univ = [r[0] for r in con.execute(f"""
        SELECT raw_symbol FROM polygon_ohlcv_1d
        WHERE raw_symbol IN (SELECT tic FROM w_comp_na_daily_all__funda
                             WHERE fyear=2021 AND sale>20000 AND datafmt='STD' AND consol='C')
          AND timestamp BETWEEN '{START}' AND '{END}'
        GROUP BY raw_symbol HAVING count(*)>1800 AND avg(close*volume)>5e7
        ORDER BY avg(close*volume) DESC LIMIT {N_UNIV}""").fetchall()]
    tl = ",".join(f"'{t}'" for t in univ)
    con.execute("SET TimeZone='UTC'")
    px = con.execute(f"""
        SELECT raw_symbol tic, date_trunc('month', timestamp) m, last(close ORDER BY timestamp) px
        FROM polygon_ohlcv_1d WHERE raw_symbol IN ({tl}) AND timestamp BETWEEN '{START}' AND '{END}'
          AND close>0 GROUP BY 1,2 ORDER BY 1,2""").fetchdf()
    px["m"] = pd.to_datetime(px["m"]).dt.strftime("%Y-%m")
    px = px.sort_values(["tic", "m"])
    px["ret_next"] = px.groupby("tic")["px"].shift(-1) / px["px"] - 1.0      # next-month return (label)
    px["ret_3m"] = px["px"] / px.groupby("tic")["px"].shift(3) - 1.0          # PIT trailing
    px["ret_12m"] = px["px"] / px.groupby("tic")["px"].shift(12) - 1.0
    # PIT latest annual revenue (as of the decision month) for context + recall flag
    rev = con.execute(f"""
        SELECT tic, CAST(fyear AS INT) fy, round(sale/1000.0,1) rev_b
        FROM w_comp_na_daily_all__funda WHERE tic IN ({tl}) AND datafmt='STD' AND consol='C'
          AND indfmt='INDL' AND popsrc='D' AND sale>0
        QUALIFY row_number() OVER (PARTITION BY tic,fyear ORDER BY sale DESC)=1""").fetchdf()
    return px.dropna(subset=["ret_next", "ret_3m"]), rev, univ


def context(row):
    return (f"{row['tic']}: trailing 3-month return {row['ret_3m']*100:+.0f}%, "
            f"trailing 12-month return {row.get('ret_12m', float('nan'))*100:+.0f}%.")


def decide_open(model, tok, tic, m, ctx):
    q = (f"You are a portfolio manager. As of the start of {m}, given:\n{ctx}\nWill {tic} OUTPERFORM or "
         f"UNDERPERFORM the average stock in your universe over the coming month? Commit to one word: "
         f"OUTPERFORM or UNDERPERFORM.")
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt").to(H.DEVICE)
    except Exception:  # noqa: BLE001
        ids = tok(q, return_tensors="pt").input_ids.to(H.DEVICE)
    import torch
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=(tok.eos_token_id or tok.pad_token_id))
    t = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).lower()
    if "outperform" in t and "underperform" not in t:
        return 1
    if "underperform" in t and "outperform" not in t:
        return -1
    if "outperform" in t and "underperform" in t:
        return 1 if t.index("outperform") < t.index("underperform") else -1
    return 0


def recall_hit_open(model, tok, tic, fy, rev):
    pre = tok.encode(f"{tic} reported total annual revenue for fiscal year {fy} of approximately $",
                     add_special_tokens=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    t = fmt(rev); dist = [x for x in sorted({fmt(rev * mu) for mu in MULTS}) if x != t and x >= 0.1]
    nt = H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{t} billion"))
    nds = [x for x in (H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{d} billion")) for d in dist) if np.isfinite(x)]
    return int(np.isfinite(nt) and nds and all(nt < x for x in nds))


def sharpe(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return float(np.mean(r) / np.std(r) * np.sqrt(12)) if len(r) > 1 and np.std(r) > 0 else float("nan")


def maxdd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def run_open(mdir, px, rev):
    name = os.path.basename(mdir.rstrip("/\\"))
    model, tok, _ = H.load_hf_causal(mdir)
    recall_cache = {}
    rows = []
    for m, day in px.groupby("m"):
        for _, r in day.iterrows():
            sig = decide_open(model, tok, r["tic"], m, context(r))
            if sig == 0:
                continue
            fy = int(m[:4]) - 1
            key = (r["tic"], fy)
            if key not in recall_cache:
                rr = rev[(rev.tic == r["tic"]) & (rev.fy == fy)]
                recall_cache[key] = recall_hit_open(model, tok, r["tic"], fy, float(rr.rev_b.iloc[0])) if len(rr) else 0
            rows.append({"m": m, "tic": r["tic"], "sig": sig, "ret_next": r["ret_next"],
                         "recall_hit": recall_cache[key], "post": m >= CUTOFF_MONTH})
    import torch; del model; torch.cuda.empty_cache()
    return name, pd.DataFrame(rows)


def analyze(name, d):
    # monthly long-short return (equal weight), and the recall-detectable sub-portfolio
    def port(df):
        g = df.groupby("m").apply(lambda x: (x.sig * x.ret_next).mean(), include_groups=False)
        return g.dropna()
    res = {"model": name, "n_decisions": int(len(d)),
           "frac_outperform": round(float((d.sig == 1).mean()), 3)}
    for lab, sub in [("all", d), ("pre_cutoff", d[~d.post]), ("post_cutoff", d[d.post]),
                     ("recall_hit", d[d.recall_hit == 1]), ("recall_miss", d[d.recall_hit == 0])]:
        p = port(sub)
        res[lab] = {"sharpe": round(sharpe(p), 3), "ann_ret": round(float(p.mean() * 12), 4),
                    "maxdd": round(maxdd(p), 3), "n_months": int(len(p)), "n_dec": int(len(sub))}
    return res


def main():
    con = duckdb.connect(str(DB), read_only=True)
    px, rev, univ = load_data(con); con.close()
    print(f"[universe] {len(univ)} names, {px.m.nunique()} months, {len(px)} name-months "
          f"({START[:7]}..{END[:7]}, cutoff {CUTOFF_MONTH})", flush=True)
    models = sys.argv[1:] or [r"E:\models\Qwen2.5-14B-Instruct", r"E:\models\Llama-3.1-8B-Instruct"]
    allres = {}
    for mdir in models:
        if not (Path(mdir) / "config.json").exists():
            print(f"  [{os.path.basename(mdir)}: not downloaded]", flush=True); continue
        name, d = run_open(mdir, px, rev)
        d.to_parquet(OUT / f"backtest_{name}.parquet", index=False)
        r = analyze(name, d); allres[name] = r
        print(f"  {name}: pre Sharpe={r['pre_cutoff']['sharpe']} post={r['post_cutoff']['sharpe']} | "
              f"recall_hit Sharpe={r['recall_hit']['sharpe']} miss={r['recall_miss']['sharpe']}", flush=True)
    (OUT / "backtest_summary.json").write_text(json.dumps(allres, indent=2))
    print(f"\n[written] {OUT/'backtest_summary.json'}")
    print("READ: leakage realises iff pre_cutoff Sharpe >> post AND recall_hit >> recall_miss.")


if __name__ == "__main__":
    main()
