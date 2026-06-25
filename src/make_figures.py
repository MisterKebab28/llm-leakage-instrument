#!/usr/bin/env python
"""make_figures.py -- publication figures from the leakage result JSONs (CPU, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "leakage"
FIGS = ROOT / "research_paper_leakage" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def load(p):
    f = OUT / p
    return json.loads(f.read_text()) if f.exists() else None


def fig_capacity():
    d = load("value_recall_cf.json")
    if not d:
        return
    bm = d["by_model"]
    items = sorted(bm.items(), key=lambda kv: kv[1]["share_true_top1"])
    names = [k for k, _ in items]
    vals = [v["share_true_top1"] for _, v in items]
    colors = ["#c0392b" if n.startswith("chrono") else "#2471a3" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color=colors)
    ax.axvline(0.20, ls="--", color="gray", label="chance (20%)")
    ax.set_xlabel("counterfactual value-recall (true-value top-1)")
    ax.set_title("Value-recall capacity: frontier (blue) vs no-leak chrono baseline (red)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_capacity.png", dpi=150); plt.close(fig)
    print("wrote fig1_capacity.png")


def fig_armB():
    files = sorted((OUT / "armB").glob("eval_*.json"))
    if not files:
        return
    rows = [json.loads(f.read_text()) for f in files]
    rows.sort(key=lambda r: -r["injected_detection_DiD_top1"])
    names = [r["model"] for r in rows]
    tb = [r["base"]["train"]["top1"] for r in rows]
    tf = [r["finetuned"]["train"]["top1"] for r in rows]
    cf = [r["finetuned"]["control"]["top1"] for r in rows]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, tb, w, label="TRAIN base", color="#aed6f1")
    ax.bar(x, tf, w, label="TRAIN fine-tuned (injected)", color="#1f618d")
    ax.bar(x + w, cf, w, label="CONTROL fine-tuned (held-out)", color="#e59866")
    ax.axhline(0.20, ls="--", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("value-recall top-1"); ax.set_ylim(0, 1.05)
    ax.set_title("Arm B positive control: injection detected on TRAIN, CONTROL unmoved")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig2_armB.png", dpi=150); plt.close(fig)
    print("wrote fig2_armB.png")


def fig_transmission():
    d = load("transmission_auc.json")
    if not d:
        return
    pts = [(v.get("value_recall_capacity"), v.get("auc_seen"), m)
           for m, v in d["per_model"].items()
           if v.get("value_recall_capacity") is not None and v.get("auc_seen") is not None]
    if not pts:
        return
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 5))
    for x, y, m in pts:
        c = "#c0392b" if m.startswith("chrono") else "#2471a3"
        ax.scatter(x, y, color=c); ax.annotate(m, (x, y), fontsize=7, xytext=(3, 3),
                                                textcoords="offset points")
    ax.axhline(0.5, ls="--", color="gray")
    corr = d.get("transmission_corr_capacity_vs_auc_seen")
    ax.set_xlabel("value-recall capacity (counterfactual top-1)")
    ax.set_ylabel("realized directional leakage  AUC_seen")
    ax.set_title(f"Recall->trading transmission  (corr = {corr})")
    fig.tight_layout(); fig.savefig(FIGS / "fig3_transmission.png", dpi=150); plt.close(fig)
    print("wrote fig3_transmission.png")


def fig_power():
    d = load("power_preregistration.json")
    if not d:
        return
    sens = d["universe_sensitivity"]
    us = list(sens.keys())
    ic = [sens[u]["detectable_abs_IC_iid"] for u in us]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(us, ic, color="#5d6d7e")
    naive = d["detectable_mean_abs_IC"]["v4_naive_breadth_N"]
    ax.axhline(naive, ls="--", color="#c0392b", label=f"v4 naive (indep) = {naive}")
    ax.set_ylabel("dependence-corrected detectable mean |IC|")
    ax.set_title("IC power: dependence correction (~3-4x the naive number)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig4_power.png", dpi=150); plt.close(fig)
    print("wrote fig4_power.png")


def fig_scale():
    d = load("value_recall_scale.json")
    if not d:
        print("fig5_scale: value_recall_scale.json not ready yet (skip)")
        return
    bm = {m: v for m, v in d["by_model"].items() if not m.startswith("chrono")}
    buckets = ["small($2-?)", "mid", "large"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bm)); w = 0.25
    for i, b in enumerate(buckets):
        vals = [v["by_size"].get(b, np.nan) for v in bm.values()]
        ax.bar(x + (i - 1) * w, vals, w, label=b)
    ax.axhline(d["chance_top1"], ls="--", color="gray", label=f"chance {d['chance_top1']}")
    ax.set_xticks(x); ax.set_xticklabels(list(bm.keys()), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("value-recall top-1"); ax.set_title(f"Scaled value-recall by firm size ({d['n_firms']} firms)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / "fig5_scale.png", dpi=150); plt.close(fig)
    print("wrote fig5_scale.png")


if __name__ == "__main__":
    fig_capacity(); fig_armB(); fig_transmission(); fig_power(); fig_scale()
    print(f"figures -> {FIGS}")
