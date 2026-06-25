#!/usr/bin/env python
"""
vr_12_crosssec.py -- value-conditioned / cross-sectional transmission (the decisive remaining test).

Directional transmission was null, but (a) it used an up/down task that base & instruct models answer
with the market base rate, and (b) realization may live in RELATIVE ranking, not absolute direction.
This tests the cross-sectional task that fixes both:

  task    : did {firm} OUTPERFORM or UNDERPERFORM the typical large-cap that year?
            ground truth = firm annual return vs the cross-sectional MEDIAN return that year (~50/50).
  recall  : value-recall instrument flags whether the model recalls the firm's revenue (in-memory).
  test    : is the model's relative call more accurate on recall-detectable cells, net of FIRM+YEAR FE?

The balanced (~50/50) label removes the base-rate degeneracy. If accuracy concentrates on recall-hit
cells (firm-FE survives) -> value-conditioned transmission (leakage realizes in ranking). If null ->
the capacity-without-realization dissociation holds for ranking too.

Run under conda `base` (GPU). Output: outputs/leakage/crosssec_cells.parquet + _fe.json
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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402
from vr_11_multicutoff_rd import panel as rd_panel, fmt  # reuse FY2013-2025 firm panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
MULTS = [0.5, 0.7, 1.4, 2.0]
# instruction-tuned (committed relative call) + a couple strong bases for the NLL variant
DEFAULT = [r"E:\models\Qwen2.5-14B-Instruct", r"E:\models\Llama-3.1-8B-Instruct",
           r"E:\models\Qwen2.5-7B-Instruct", r"E:\models\Mistral-Nemo-Instruct-2407"]


def build_panel(con):
    cells = rd_panel(con, int(os.environ.get("BARJ_NFIRMS", "150")))  # tic, fy, name, rev_b + returns
    # add annual return + cross-sectional median -> relative label
    con.execute("SET TimeZone='UTC'")
    tl = ",".join(f"'{t}'" for t in cells.tic.unique())
    px = con.execute(
        f"""WITH p AS (SELECT raw_symbol, CAST(timestamp AS DATE) d, close, year(timestamp) yr
                       FROM polygon_ohlcv_1d WHERE raw_symbol IN ({tl}) AND close>0 AND volume>0
                         AND NOT (month(timestamp)=1 AND day(timestamp)=1))
            SELECT raw_symbol tic, yr fy, arg_min(close,d) c0, arg_max(close,d) c1
            FROM p GROUP BY 1,2 HAVING count(*)>=150""").fetchdf()
    px["ret"] = px["c1"] / px["c0"] - 1.0
    px["med"] = px.groupby("fy")["ret"].transform("median")
    px["outperf"] = np.where(px["ret"] > px["med"], "outperform", "underperform")
    return cells.merge(px[["tic", "fy", "outperf"]], on=["tic", "fy"], how="inner")


def decide(model, tok, nm, fy, instruct, n_new=16):
    q = (f"During the {fy} calendar year, did {nm} stock OUTPERFORM or UNDERPERFORM the typical "
         f"large-cap US stock? You must commit. Reply with exactly one word: OUTPERFORM or UNDERPERFORM.")
    if instruct:
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt").to(H.DEVICE)
        except Exception:  # noqa: BLE001
            ids = tok(q, return_tensors="pt").input_ids.to(H.DEVICE)
        import torch
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=(tok.eos_token_id or tok.pad_token_id))
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).lower()
    else:
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        pre = tok.encode(q + " ", add_special_tokens=True)
        no = H.value_recall_nll(model, "hf-causal", tok, pre, enc("OUTPERFORM"))
        nu = H.value_recall_nll(model, "hf-causal", tok, pre, enc("UNDERPERFORM"))
        return "outperform" if no < nu else "underperform"
    if "outperform" in txt and "underperform" not in txt:
        return "outperform"
    if "underperform" in txt and "outperform" not in txt:
        return "underperform"
    if "outperform" in txt and "underperform" in txt:
        return "outperform" if txt.index("outperform") < txt.index("underperform") else "underperform"
    return None


def recall_hit(model, tok, nm, fy, rev):
    pre = tok.encode(f"{nm} reported total annual revenue for fiscal year {fy} of approximately $",
                     add_special_tokens=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    t = fmt(rev); dist = [x for x in sorted({fmt(rev * m) for m in MULTS}) if x != t and x >= 0.1]
    nt = H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{t} billion"))
    nds = [H.value_recall_nll(model, "hf-causal", tok, pre, enc(f"{d} billion")) for d in dist]
    nds = [x for x in nds if np.isfinite(x)]
    return int(np.isfinite(nt) and nds and all(nt < x for x in nds))


def two_way_fe(d, ycol):
    y = d[ycol].astype(float).reset_index(drop=True)
    x = d["recall_hit"].astype(float).reset_index(drop=True)
    tic = d["tic"].reset_index(drop=True); fy = d["fy"].reset_index(drop=True)
    for _ in range(30):
        y = (y - y.groupby(tic).transform("mean")) ; y = y - y.groupby(fy).transform("mean")
        x = (x - x.groupby(tic).transform("mean")) ; x = x - x.groupby(fy).transform("mean")
    sxx = float((x * x).sum())
    if sxx < 1e-9:
        return None
    beta = float((x * y).sum() / sxx); e = y - beta * x
    meat = float(sum((x * e).groupby(tic).sum() ** 2)); se = float(np.sqrt(meat) / sxx)
    t = beta / se if se > 0 else float("nan"); nclu = tic.nunique()
    return {"beta": round(beta, 4), "se": round(se, 4), "t": round(t, 2),
            "p": round(float(2 * stats.t.sf(abs(t), df=max(1, nclu - 1))), 4), "n": int(len(d))}


def main():
    models = sys.argv[1:] or DEFAULT
    con = duckdb.connect(str(DB), read_only=True)
    cells = build_panel(con); con.close()
    print(f"[panel] {len(cells)} cells, outperform base-rate={(cells.outperf=='outperform').mean():.2f}",
          flush=True)
    res = {}
    for mdir in models:
        name = os.path.basename(mdir.rstrip("/\\"))
        instruct = "Instruct" in name or "-it" in name
        if not (Path(mdir) / "config.json").exists():
            print(f"  [{name}: not downloaded]", flush=True); continue
        try:
            model, tok, _ = H.load_hf_causal(mdir)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name} load failed: {str(e).splitlines()[0][:60]}]", flush=True); continue
        rows = []
        for _, f in cells.iterrows():
            dec = decide(model, tok, f["name"], int(f["fy"]), instruct)
            if dec is None:
                continue
            rows.append({"tic": f["tic"], "fy": int(f["fy"]),
                         "recall_hit": recall_hit(model, tok, f["name"], int(f["fy"]), float(f["rev_b"])),
                         "decision": dec, "correct": int(dec == f["outperf"])})
        import torch; del model; torch.cuda.empty_cache()
        d = pd.DataFrame(rows)
        if len(d) < 50:
            print(f"  [{name}: {len(d)} decisions, skip]", flush=True); continue
        d.assign(model=name).to_parquet(OUT / f"crosssec_cells_{name}.parquet", index=False)
        fe = two_way_fe(d, "correct")
        res[name] = {"n": int(len(d)), "frac_outperform": round(float((d.decision == "outperform").mean()), 3),
                     "overall_acc": round(float(d.correct.mean()), 3),
                     "recall_rate": round(float(d.recall_hit.mean()), 3),
                     "acc_hit": round(float(d[d.recall_hit == 1].correct.mean()), 3) if (d.recall_hit == 1).any() else None,
                     "acc_miss": round(float(d[d.recall_hit == 0].correct.mean()), 3) if (d.recall_hit == 0).any() else None,
                     "twoway_firm_year_FE": fe}
        r = res[name]
        print(f"  {name}: frac_out={r['frac_outperform']} acc={r['overall_acc']} "
              f"hit={r['acc_hit']} miss={r['acc_miss']} FE {fe}", flush=True)
    out = {"design": "cross-sectional relative (outperform median) ~ recall_hit + firm+year FE; balanced label",
           "by_model": res,
           "interpretation": "FE beta>0, p<0.05 => value-conditioned transmission (leakage realizes in "
                             "relative ranking). Null => dissociation holds for ranking too."}
    (OUT / "crosssec_fe.json").write_text(json.dumps(out, indent=2))
    print(f"\n[written] {OUT/'crosssec_fe.json'}")


if __name__ == "__main__":
    main()
