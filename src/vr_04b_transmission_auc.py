#!/usr/bin/env python
"""
vr_04b_transmission_auc.py -- bias-immune re-analysis of the transmission probe.

The accuracy metric in vr_04 is confounded by each model's directional bias x the
base rate (stocks mostly rise). The clean signal is whether the model's continuous
lean (`lean_higher_margin` = NLL_lower - NLL_higher) DISCRIMINATES true up- from
down-years -- i.e. AUC(margin, true_dir==higher). A constant "higher" lean cancels
in AUC, so this isolates recall from bias.

  AUC_seen   : discrimination on memorisable (year <= cutoff) outcomes
  AUC_unseen : discrimination on never-seen (year > cutoff) outcomes
  realized_leakage = AUC_seen - 0.5  (and AUC_seen - AUC_unseen where both exist)
  TRANSMISSION = corr across models of value-recall capacity (vr_03) vs realized_leakage.

Reads outputs/leakage/transmission.parquet (no model re-run). Output: transmission_auc.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score|pos > score|neg). labels in {0,1}."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    df = pd.read_parquet(OUT / "transmission.parquet")
    df = df.assign(y=(df["true_dir"] == "higher").astype(int))
    cap = {}
    cf = OUT / "value_recall_cf.json"
    if cf.exists():
        cap = {m: v["share_true_top1"] for m, v in json.load(open(cf))["by_model"].items()}

    per = {}
    for m, d in df.groupby("model"):
        s, u = d[d.seen == 1], d[d.seen == 0]
        auc_s = auc(s["lean_higher_margin"].to_numpy(), s["y"].to_numpy()) if len(s) else float("nan")
        auc_u = auc(u["lean_higher_margin"].to_numpy(), u["y"].to_numpy()) if len(u) else float("nan")
        per[m] = {
            "auc_seen": None if np.isnan(auc_s) else round(auc_s, 3),
            "n_seen": int(len(s)), "n_seen_down": int((s.y == 0).sum()),
            "auc_unseen": None if np.isnan(auc_u) else round(auc_u, 3),
            "n_unseen": int(len(u)), "n_unseen_down": int((u.y == 0).sum()),
            "realized_leakage_auc_seen_minus_0.5": None if np.isnan(auc_s) else round(auc_s - 0.5, 3),
            "value_recall_capacity": cap.get(m),
            "is_baseline": m.startswith("chrono"),
        }

    # transmission: capacity vs (AUC_seen - 0.5), across models with both defined
    pts = [(v["value_recall_capacity"], v["realized_leakage_auc_seen_minus_0.5"])
           for v in per.values()
           if v["value_recall_capacity"] is not None and v["realized_leakage_auc_seen_minus_0.5"] is not None]
    corr = None
    if len(pts) >= 3:
        a = np.array(pts)
        if a[:, 0].std() > 0 and a[:, 1].std() > 0:
            corr = round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3)

    # group means: frontier (capacity-bearing) vs no-leak baseline
    front = [v["auc_seen"] for v in per.values() if not v["is_baseline"] and v["auc_seen"] is not None]
    base = [v["auc_seen"] for v in per.values() if v["is_baseline"] and v["auc_seen"] is not None]
    summary = {
        "metric": "AUC(lean_higher_margin, true_dir==higher) -- bias-immune directional recall",
        "per_model": per,
        "frontier_mean_auc_seen": round(float(np.mean(front)), 3) if front else None,
        "baseline_mean_auc_seen": round(float(np.mean(base)), 3) if base else None,
        "transmission_corr_capacity_vs_auc_seen": corr,
        "interpretation": (
            "AUC_seen>0.5 => the model's directional lean discriminates true up/down on "
            "memorisable years = recall realized in the decision. Frontier mean vs baseline "
            "mean isolates the capacity effect; transmission_corr>0 means capacity predicts "
            "realized directional leakage across models. Caveat: thin/again-confounded unseen "
            "arm (recent cutoffs, v4 power limit) -- definitive RD needs forward accrual."),
    }
    (OUT / "transmission_auc.json").write_text(json.dumps(summary, indent=2))

    print("=== bias-immune transmission (AUC of directional lean) ===")
    for m in sorted(per, key=lambda k: -(per[k]["auc_seen"] or 0)):
        v = per[m]
        print(f"  {m:18s} AUC_seen={v['auc_seen']} (n{v['n_seen']},dn{v['n_seen_down']}) "
              f"AUC_unseen={v['auc_unseen']} cap={v['value_recall_capacity']}")
    print(f"\nfrontier mean AUC_seen={summary['frontier_mean_auc_seen']} vs "
          f"baseline {summary['baseline_mean_auc_seen']}")
    print(f"TRANSMISSION corr(capacity, AUC_seen-0.5) = {corr}")
    print(f"[written] {OUT/'transmission_auc.json'}")


if __name__ == "__main__":
    main()
