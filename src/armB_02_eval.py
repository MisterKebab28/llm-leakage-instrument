#!/usr/bin/env python
"""
armB_02_eval.py -- Arm B positive-control evaluation (instrument POWER / sensitivity).

Counterfactual value-recall (vr_03-style: true value vs {0.5,0.7,1.4,2.0}x distractors)
on TRAIN vs CONTROL firms, for the BASE model and the LoRA-FINETUNED model. The eval
prompt differs from the injection templates, so this tests genuine recall, not string copy.

Positive control (the DiD):
  injected_detection = [top1_train_post - top1_train_pre] - [top1_control_post - top1_control_pre]
A large positive DiD = fine-tuning injected leakage on TRAIN and the instrument detects it,
while CONTROL stays at baseline = the value-recall detector has power. Also reports an
FSD-style member/non-member NLL gap.

Runs under `leak_ft`. Usage: armB_02_eval.py <base_dir> <adapter_dir>
Output: outputs/leakage/armB/eval_<name>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_harness as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "leakage" / "armB"
MULTS = [0.5, 0.7, 1.4, 2.0]


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def candidates(rev_b):
    true = fmt(rev_b)
    dist = []
    for m in MULTS:
        d = fmt(rev_b * m)
        if d != true and d not in dist and (d if isinstance(d, int) else d) >= 0.1:
            dist.append(d)
    return true, dist


def eval_model(model, tok, facts):
    rows = []
    for _, f in facts.iterrows():
        pre = tok.encode(f"{f['name']} reported total annual revenue for fiscal year "
                         f"{int(f['fy'])} of approximately $", add_special_tokens=True)
        true, dist = candidates(f["rev_b"])
        nt = H.value_recall_nll(model, "hf-causal", tok, pre,
                                tok.encode(f"{true} billion", add_special_tokens=False))
        nds = [H.value_recall_nll(model, "hf-causal", tok, pre,
                                  tok.encode(f"{d} billion", add_special_tokens=False)) for d in dist]
        nds = [x for x in nds if np.isfinite(x)]
        if not nds or not np.isfinite(nt):
            continue
        rows.append({"split": f["split"], "nll_true": nt,
                     "margin": float(np.mean(nds) - nt),
                     "top1": int(all(nt < x for x in nds))})
    return pd.DataFrame(rows)


def summarize(df, tag):
    out = {}
    for sp in ("train", "control"):
        d = df[df.split == sp]
        out[sp] = {"top1": round(float(d.top1.mean()), 3),
                   "mean_margin": round(float(d.margin.mean()), 3),
                   "mean_nll_true": round(float(d.nll_true.mean()), 3), "n": int(len(d))}
    print(f"  [{tag}] train top1={out['train']['top1']} margin={out['train']['mean_margin']} | "
          f"control top1={out['control']['top1']} margin={out['control']['mean_margin']}", flush=True)
    return out


def main():
    base, adapter = sys.argv[1], sys.argv[2]
    name = os.path.basename(base.rstrip("/\\"))
    import duckdb  # robust parquet read (avoids a pyarrow 19 repetition-histogram read bug)
    facts = duckdb.sql(f"SELECT * FROM '{(OUT / 'facts.parquet').as_posix()}'").df()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    kw = {} if os.path.isdir(base) else {"cache_dir": r"E:\hf_cache"}
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True, **kw).to("cuda").eval()

    print(f"[{name}] evaluating BASE (pre-injection)...", flush=True)
    pre = summarize(eval_model(model, tok, facts), "base")

    print(f"[{name}] attaching LoRA adapter + evaluating FINETUNED (post-injection)...", flush=True)
    ft = PeftModel.from_pretrained(model, adapter).eval()
    post = summarize(eval_model(ft, tok, facts), "finetuned")

    did = ((post["train"]["top1"] - pre["train"]["top1"])
           - (post["control"]["top1"] - pre["control"]["top1"]))
    fsd = ((pre["control"]["mean_nll_true"] - post["control"]["mean_nll_true"])  # control drift
           )
    res = {
        "model": name, "base": pre, "finetuned": post,
        "injected_detection_DiD_top1": round(float(did), 3),
        "train_top1_jump": round(post["train"]["top1"] - pre["train"]["top1"], 3),
        "control_top1_jump": round(post["control"]["top1"] - pre["control"]["top1"], 3),
        "fsd_member_nll_drop": round(pre["train"]["mean_nll_true"] - post["train"]["mean_nll_true"], 3),
        "fsd_nonmember_nll_drop": round(fsd, 3),
        "interpretation": ("DiD>0 and train_top1_jump>>control_top1_jump => fine-tuning injected "
                           "leakage on TRAIN and the value-recall instrument detects it while CONTROL "
                           "stays at baseline = the detector has POWER (sensitivity). The FSD member "
                           "NLL drop >> non-member confirms membership at the loss level."),
    }
    (OUT / f"eval_{name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{name}] DiD(top1)={res['injected_detection_DiD_top1']}  "
          f"train_jump={res['train_top1_jump']}  control_jump={res['control_top1_jump']}  "
          f"FSD member_drop={res['fsd_member_nll_drop']} vs nonmember={res['fsd_nonmember_nll_drop']}")
    print(f"[written] {OUT/('eval_'+name+'.json')}")


if __name__ == "__main__":
    main()
