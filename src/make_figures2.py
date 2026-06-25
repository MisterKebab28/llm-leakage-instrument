#!/usr/bin/env python
"""Branch-B figures: precise-capacity, specificity (wide/tight/gen), transmission-null confound."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"
FIG = Path(__file__).resolve().parents[2] / "research_paper_leakage" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

spec = json.load(open(OUT / "specificity.json"))["by_model"]
try:
    g4 = json.load(open(OUT / "gemma4_gemma-4-12B.json"))
    spec["gemma-4-12B"] = {"wide_top1": g4["wide_top1"], "tight_top1": g4["tight_top1"],
                           "gen_within_10pct": g4["gen_within_10pct"], "is_baseline": False}
except FileNotFoundError:
    pass

front = {m: v for m, v in spec.items() if not v["is_baseline"]}
chrono = {m: v for m, v in spec.items() if v["is_baseline"]}

# Fig 1 -- precise capacity: tight-distractor recall (magnitude-robust), sorted
order = sorted(front, key=lambda m: -front[m]["tight_top1"])
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(range(len(order)), [front[m]["tight_top1"] for m in order], color="#2c6fbb", label="frontier")
ax.bar(range(len(order), len(order) + len(chrono)), [chrono[m]["tight_top1"] for m in chrono],
       color="#c0392b", label="no-leak baseline (ChronoGPT)")
ax.axhline(0.20, ls="--", c="k", lw=1, label="chance (0.20)")
ax.set_xticks(range(len(order) + len(chrono)))
ax.set_xticklabels(order + list(chrono), rotation=60, ha="right", fontsize=7)
ax.set_ylabel("tight-distractor recall (±15%)"); ax.legend(fontsize=8)
ax.set_title("Precise value-recall capacity: magnitude-robust (tight ±15%) distractors")
fig.tight_layout(); fig.savefig(FIG / "fig1_capacity_precise.png", dpi=150); plt.close(fig)

# Fig 2 -- specificity: wide vs tight vs free-gen for top models
top = sorted(front, key=lambda m: -front[m]["wide_top1"])[:12]
x = np.arange(len(top)); w = 0.27
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w, [front[m]["wide_top1"] for m in top], w, label="wide distractors (MCQ)", color="#7fb3d5")
ax.bar(x, [front[m]["tight_top1"] for m in top], w, label="tight ±15% (magnitude-robust)", color="#2c6fbb")
ax.bar(x + w, [front[m]["gen_within_10pct"] for m in top], w, label="free generation ≤10%", color="#1a5276")
ax.axhline(0.20, ls="--", c="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(top, rotation=55, ha="right", fontsize=7)
ax.set_ylabel("true-value recall"); ax.legend(fontsize=8)
ax.set_title("Memorization vs estimation: recall survives tight distractors & free generation")
fig.tight_layout(); fig.savefig(FIG / "fig2_specificity.png", dpi=150); plt.close(fig)

# Fig 3 -- transmission null: the prominence confound (year-FE vs two-way firm+year FE)
try:
    fe = json.load(open(OUT / "agentic_panel_fe.json"))
    models = [m for m in fe if fe[m]["twoway_firm_year_FE"]]
    yfe = [fe[m]["diff_yearFE_only"] for m in models]
    tfe = [fe[m]["twoway_firm_year_FE"]["beta_twowayFE"] for m in models]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, yfe, w, label="year-FE only (apparent)", color="#e67e22")
    ax.bar(x + w / 2, tfe, w, label="+ firm FE (confound removed)", color="#2c6fbb")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("recall→accuracy effect")
    ax.set_title("Transmission is a prominence artifact: the apparent signal vanishes under firm FE")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_transmission_null.png", dpi=150); plt.close(fig)
    print("fig3 written")
except FileNotFoundError:
    print("fig3 skipped (no agentic_panel_fe.json)")
print("figures written to", FIG)
