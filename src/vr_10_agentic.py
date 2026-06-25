#!/usr/bin/env python
"""
vr_10_agentic.py -- the DECISIVE transmission test: instruction-tuned models making REAL decisions.

vr_09's within-model decomposition was null, but it used BASE models that default to the market
base rate (down-year accuracy ~0%) -> the probe couldn't detect transmission from any source. This
re-runs the SAME decomposition with INSTRUCTION-TUNED models that make actual, varied buy/hold-style
directional forecasts via their chat template:

  decision   = model's forward call (HIGHER/LOWER) for (firm, year), chat-templated, generated, parsed
  recall_hit = does the value-recall instrument detect the model recalls the firm's revenue? (in-memory)
  correct    = decision matches the realized split-adjusted return sign
  Transmission = is `correct` higher on recall_hit=1 cells, net of YEAR fixed effects, at N=thousands?

Reports the decision distribution (frac HIGHER) so we can see the decisions are non-degenerate -- the
prerequisite for the test to have power. recall (revenue) != decision (return dir) => non-circular.

Run under conda `base` (GPU). Usage: vr_10_agentic.py [model_dir ...]
Output: outputs/leakage/agentic_transmission.json
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
from vr_09_decomposition import panel, fmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
DEVICE = "cuda"
MULTS = [0.5, 0.7, 1.4, 2.0]
DEFAULT = [r"E:\models\Qwen2.5-14B-Instruct", r"E:\models\Llama-3.1-8B-Instruct",
           r"E:\models\Mistral-Nemo-Instruct-2407", r"E:\models\Qwen2.5-7B-Instruct",
           r"E:\models\phi-4"]


def decide(model, tok, nm, fy, n_new=16):
    msg = [{"role": "user", "content":
            f"It is the start of {fy}. As an equity analyst making a genuine forecast, predict whether "
            f"{nm} stock will finish the {fy} calendar year HIGHER or LOWER than where it started. "
            f"You must commit. Reply with exactly one word: HIGHER or LOWER."}]
    try:
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    except Exception:  # noqa: BLE001  (model without a chat template -> plain prompt)
        ids = tok(msg[0]["content"], return_tensors="pt").input_ids.to(DEVICE)
    import torch
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                             pad_token_id=(tok.eos_token_id or tok.pad_token_id))
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).lower()
    if "higher" in txt and "lower" not in txt:
        return "higher"
    if "lower" in txt and "higher" not in txt:
        return "lower"
    if "higher" in txt and "lower" in txt:           # both -> take first mentioned
        return "higher" if txt.index("higher") < txt.index("lower") else "lower"
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


def main():
    models = sys.argv[1:] or DEFAULT
    con = duckdb.connect(str(DB), read_only=True)
    cells = panel(con)
    con.close()
    print(f"[panel] {len(cells)} cells, base-rate higher={(cells.true_dir=='higher').mean():.2f}", flush=True)
    from scipy import stats
    results = {}
    for mdir in models:
        name = os.path.basename(mdir.rstrip("/\\"))
        if not (Path(mdir) / "config.json").exists():
            print(f"  [{name}: not downloaded, skip]", flush=True); continue
        try:
            model, tok, _ = H.load_hf_causal(mdir)
        except Exception as e:  # noqa: BLE001
            print(f"  [{name} load failed: {str(e).splitlines()[0][:70]}]", flush=True); continue
        rows = []
        for _, f in cells.iterrows():
            nm, fy, rev = f["name"], int(f["fy"]), float(f["rev_b"])
            dec = decide(model, tok, nm, fy)
            if dec is None:
                continue
            rows.append({"tic": f["tic"], "fy": fy, "recall_hit": recall_hit(model, tok, nm, fy, rev),
                         "decision": dec, "correct": int(dec == f["true_dir"]), "true_dir": f["true_dir"]})
        import torch; del model; torch.cuda.empty_cache()
        d = pd.DataFrame(rows)
        if len(d) < 50:
            print(f"  [{name}: only {len(d)} parsed decisions, skip]", flush=True); continue
        d.assign(model=name).to_parquet(OUT / f"agentic_cells_{name}.parquet", index=False)  # for firm-FE
        d["correct_fe"] = d["correct"] - d.groupby("fy")["correct"].transform("mean")
        g1, g0 = d[d.recall_hit == 1], d[d.recall_hit == 0]
        tt = stats.ttest_ind(g1["correct_fe"], g0["correct_fe"], equal_var=False) if len(g1) and len(g0) else None
        results[name] = {
            "n_cells": int(len(d)), "frac_decided_higher": round(float((d.decision == "higher").mean()), 3),
            "decision_entropy_ok": bool(0.05 < (d.decision == "higher").mean() < 0.95),
            "overall_acc": round(float(d.correct.mean()), 3),
            "recall_rate": round(float(d.recall_hit.mean()), 3),
            "acc_recall_hit": round(float(g1.correct.mean()), 3) if len(g1) else None,
            "acc_recall_miss": round(float(g0.correct.mean()), 3) if len(g0) else None,
            "acc_diff_yearFE": round(float(g1["correct_fe"].mean() - g0["correct_fe"].mean()), 3) if len(g1) and len(g0) else None,
            "t_stat": round(float(tt.statistic), 2) if tt else None,
            "p_value": round(float(tt.pvalue), 4) if tt else None,
            "down_year_acc_hit": round(float(g1[g1.true_dir == "lower"].correct.mean()), 3) if len(g1[g1.true_dir == "lower"]) else None,
            "down_year_acc_miss": round(float(g0[g0.true_dir == "lower"].correct.mean()), 3) if len(g0[g0.true_dir == "lower"]) else None,
        }
        r = results[name]
        print(f"  {name}: decided_higher={r['frac_decided_higher']} (varied={r['decision_entropy_ok']}) "
              f"acc|hit={r['acc_recall_hit']} miss={r['acc_recall_miss']} FEdiff={r['acc_diff_yearFE']} "
              f"t={r['t_stat']} p={r['p_value']} N={r['n_cells']}", flush=True)

    out = {"design": "instruction-tuned agentic directional forecast; correct ~ recall_hit + year FE",
           "note": ("frac_decided_higher in (0.05,0.95) => decisions non-degenerate (the probe has power, "
                    "unlike base models in vr_09). acc_diff_yearFE>0 with small p => transmission: recall "
                    "capacity makes real decisions more accurate = the leakage->trading link, powered."),
           "by_model": results}
    (OUT / "agentic_transmission.json").write_text(json.dumps(out, indent=2))
    print(f"\n[written] {OUT/'agentic_transmission.json'}")


if __name__ == "__main__":
    main()
