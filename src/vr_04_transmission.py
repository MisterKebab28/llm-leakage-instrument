#!/usr/bin/env python
"""
vr_04_transmission.py -- the recall->trading TRANSMISSION estimand (v5 sections 1-2).

v5's novel increment over "LLMs memorise values": does leak-CAPACITY (value-recall)
become REALIZED contamination in a decision? We operationalise the simplest possible
trade -- a directional bet -- and run the metric-level RD against the no-leak baseline.

Task: "Over calendar year Y, did shares of <Company> finish HIGHER or LOWER than they
started?" Ground truth = sign of the annual return from Polygon split-adjusted close
(single vendor, v3 P1). For (model, name, year) we score NLL(' higher') vs NLL(' lower')
and take the lower-NLL answer.

Identification:
  seen  = year <= model cutoff (outcome was memorisable)
  unseen= year >  model cutoff (outcome cannot have been seen)
  realized_leakage(model) = acc(seen) - acc(unseen)            # directional "trading jump"
  metric-level RD vs no-leak base = [acc_LLM_seen - acc_base_seen]
                                   - [acc_LLM_unseen - acc_base_unseen]
  TRANSMISSION = corr across models between value-recall capacity (vr_03 top1)
                 and realized_leakage. This cross-model link IS the result.

Down-year accuracy is reported separately: stocks rise most years, so a "higher"-biased
model scores high trivially; correctly recalling a DOWN year defeats that prior and is
the clean memorisation signal. The seen-vs-unseen gap also controls for any constant bias.

Run under conda `base` (GPU). Output: outputs/leakage/transmission.{parquet,json}
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
from vr_02_value_recall import discover_chrono, discover_frontier, FRONTIER_CUTOFF  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"

# famous, recognisable large-caps (models "know" them); news-style names
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet",
         "META": "Meta Platforms", "NVDA": "Nvidia", "TSLA": "Tesla", "NFLX": "Netflix",
         "INTC": "Intel", "JPM": "JPMorgan Chase", "WMT": "Walmart", "JNJ": "Johnson & Johnson",
         "KO": "Coca-Cola", "DIS": "Walt Disney", "BA": "Boeing", "XOM": "Exxon Mobil",
         "BAC": "Bank of America", "GE": "General Electric", "F": "Ford", "PFE": "Pfizer",
         "CSCO": "Cisco", "ORCL": "Oracle", "IBM": "IBM", "AMD": "AMD", "QCOM": "Qualcomm",
         "PYPL": "PayPal", "SBUX": "Starbucks", "NKE": "Nike", "MCD": "McDonald's", "GS": "Goldman Sachs"}
YEARS = list(range(2017, 2026))   # 2017..2025 (Polygon complete years; ends 2026-02)


def annual_direction(con) -> pd.DataFrame:
    """Sign of each (ticker, year) annual return from Polygon split-adjusted close."""
    tics = ",".join(f"'{t}'" for t in NAMES)
    df = con.execute(
        f"""
        WITH px AS (
            SELECT raw_symbol, CAST(timestamp AS DATE) d, close,
                   year(timestamp) yr
            FROM polygon_ohlcv_1d
            WHERE raw_symbol IN ({tics}) AND close > 0 AND volume > 0
              AND year(timestamp) BETWEEN {YEARS[0]} AND {YEARS[-1]}
              -- drop spurious New-Year holiday prints (never a US session; tz-cast artifact)
              AND NOT (month(timestamp) = 1 AND day(timestamp) = 1)
        ),
        bnd AS (
            SELECT raw_symbol, yr,
                   arg_min(close, d) AS c_first, arg_max(close, d) AS c_last,
                   count(*) n
            FROM px GROUP BY raw_symbol, yr
        )
        SELECT raw_symbol AS tic, yr, c_first, c_last, n,
               (c_last/c_first - 1.0) AS annual_ret
        FROM bnd WHERE n >= 200
        ORDER BY raw_symbol, yr
        """
    ).fetchdf()
    df["true_dir"] = np.where(df["annual_ret"] >= 0, "higher", "lower")
    df["name"] = df["tic"].map(NAMES)
    return df


def prompt_for(name, yr):
    return (f"Over the {yr} calendar year, did shares of {name} finish higher or lower "
            f"than where they started? Answer with one word:")


def score(model, kind, tok, name, yr, hf=False):
    enc = (lambda s: tok.encode(s, add_special_tokens=False)) if hf else tok.encode
    pre = tok.encode(prompt_for(name, yr), add_special_tokens=True) if hf else tok.encode(prompt_for(name, yr))
    nh = H.value_recall_nll(model, kind, tok, pre, enc(" higher"))
    nl = H.value_recall_nll(model, kind, tok, pre, enc(" lower"))
    if not (np.isfinite(nh) and np.isfinite(nl)):
        return None
    pred = "higher" if nh < nl else "lower"
    return pred, float(nl - nh)   # margin>0 => model leans 'higher'


def run(name, loader, facts, cutoff, kind, hf):
    out = loader()
    model, tok = (out[0], out[1]) if hf else (out[0], H.gpt2_tokenizer())
    rows = []
    for _, f in facts.iterrows():
        r = score(model, kind, tok, f["name"], int(f["yr"]), hf=hf)
        if r:
            pred, margin = r
            rows.append({"model": name, "cutoff": cutoff, "tic": f["tic"], "yr": int(f["yr"]),
                         "true_dir": f["true_dir"], "pred": pred,
                         "correct": int(pred == f["true_dir"]),
                         "seen": int(f["yr"] <= cutoff), "lean_higher_margin": margin})
    import torch; del model; torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def acc(df):
    return float(df["correct"].mean()) if len(df) else float("nan")


def main():
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET TimeZone='UTC'")   # deterministic date casting (avoid tz-dependent day shifts)
    facts = annual_direction(con)
    con.close()
    base = (facts["true_dir"] == "higher").mean()
    print(f"[facts] {len(facts)} (name,year) cells, {facts['tic'].nunique()} names, "
          f"{facts['yr'].nunique()} years; base rate higher={base:.2f}")

    parts = []
    for repo in discover_chrono():
        cut = int(repo.rsplit("-", 1)[-1][:4])
        parts.append(run(f"chrono-{cut}", lambda r=repo: H.load_chrono_gpt(r), facts, cut, "chrono-gpt", False))
        print(f"  scored chrono-{cut}")
    for fpath in discover_frontier():
        nm = os.path.basename(fpath)
        cut = FRONTIER_CUTOFF.get(nm, 2024)
        try:
            parts.append(run(nm, lambda p=fpath: H.load_hf_causal(p), facts, cut, "hf-causal", True))
            print(f"  scored {nm}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{nm} failed: {str(e).splitlines()[0][:90]}]")

    res = pd.concat([p for p in parts if len(p)], ignore_index=True)
    res.to_parquet(OUT / "transmission.parquet", index=False)

    # value-recall capacity per model (from vr_03) for the transmission link
    cap = {}
    cf = OUT / "value_recall_cf.json"
    if cf.exists():
        cap = {m: v["share_true_top1"] for m, v in json.load(open(cf))["by_model"].items()}

    per = {}
    for m, d in res.groupby("model"):
        seen, unseen = d[d.seen == 1], d[d.seen == 0]
        down = d[d.true_dir == "lower"]
        per[m] = {
            "acc_all": round(acc(d), 3), "n": int(len(d)),
            "acc_seen": round(acc(seen), 3), "n_seen": int(len(seen)),
            "acc_unseen": round(acc(unseen), 3) if len(unseen) else None, "n_unseen": int(len(unseen)),
            "acc_down_years": round(acc(down), 3), "n_down": int(len(down)),
            "acc_down_seen": round(acc(down[down.seen == 1]), 3) if len(down[down.seen == 1]) else None,
            "realized_leakage_seen_minus_unseen": (round(acc(seen) - acc(unseen), 3)
                                                   if len(unseen) else None),
            "value_recall_capacity": cap.get(m),
        }

    # transmission: corr(capacity, realized leakage) across models that have both + unseen cells
    pts = [(v["value_recall_capacity"], v["realized_leakage_seen_minus_unseen"])
           for v in per.values()
           if v["value_recall_capacity"] is not None and v["realized_leakage_seen_minus_unseen"] is not None]
    transmission_corr = None
    if len(pts) >= 3:
        a = np.array(pts)
        transmission_corr = round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3)

    summary = {"task": "annual price-direction recall (higher/lower)", "years": YEARS,
               "n_names": len(NAMES), "base_rate_higher": round(float(base), 3),
               "per_model": per, "transmission_corr_capacity_vs_realized_leakage": transmission_corr,
               "interpretation": (
                   "Frontier models with high value-recall capacity should show acc_seen >> 0.5 "
                   "(esp. acc_down_seen, which defeats the up-bias) and acc_unseen ~ chance; the "
                   "no-leak baseline ~chance throughout. A positive transmission_corr means "
                   "memorisation capacity is realized as directional decisions = leakage drives "
                   "the 'trade', not skill. NOTE: frontier cutoffs are recent so n_unseen is thin "
                   "(the v4 post-cutoff power limit) -- read acc_unseen with that caveat.")}
    (OUT / "transmission.json").write_text(json.dumps(summary, indent=2))

    print("\n=== recall->trading transmission (annual direction) ===")
    print(f"base rate higher = {base:.2f}\n")
    for m in sorted(per, key=lambda k: -(per[k]['acc_seen'] or 0)):
        v = per[m]
        print(f"  {m:18s} seen={v['acc_seen']:.2f}(n{v['n_seen']}) "
              f"unseen={v['acc_unseen'] if v['acc_unseen'] is not None else 'NA'}(n{v['n_unseen']}) "
              f"down_seen={v['acc_down_seen']} cap={v['value_recall_capacity']}")
    print(f"\nTRANSMISSION corr(capacity, realized_leakage) = {transmission_corr}")
    print(f"[written] {OUT/'transmission.json'}")


if __name__ == "__main__":
    main()
