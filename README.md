# A Validated Instrument for Memorization Leakage in LLM Trading Evaluation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20844335.svg)](https://doi.org/10.5281/zenodo.20844335)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reference implementation and reproduction artifact for the paper
*"Memorized but Not Realized: Frontier Language Models Precisely Recall Financial Fundamentals,
but It Does Not Become Trading Skill."*

When a large language model is evaluated on data that predates its training cutoff, a strong result may
reflect **recall of the realised outcome** rather than genuine foresight. This repository provides a
**calibrated instrument** that measures such leakage, a **validation cascade** that certifies the
instrument is both specific and sensitive, and the full pipeline reproducing every figure and number in
the paper.

The central distinction the instrument makes precise is between **capacity** (does the model *know* the
outcome?) and **realization** (does that knowledge *change its decision*?). Our headline empirical
finding is that the two come apart: frontier models memorize financial fundamentals precisely, but that
memory does not transmit to trading skill.

## The instrument

For a fact `(entity, date, realised value V)`, the model is scored on its preference for `V` over
counterfactuals:

| probe | construction | isolates |
|---|---|---|
| **wide** | distractors `{0.5, 0.7, 1.4, 2.0}×V` | recall vs. random |
| **tight** | log-matched distractors `{0.85, 0.93, 1.08, 1.18}×V` | *precise* recall vs. order-of-magnitude estimation |
| **free generation** | no options shown; model writes the number, scored within 5/10/20% | recall without recognition |

White-box, candidates are ranked by summed log-probability (`vr_harness.value_recall_nll`); black-box,
they are presented as multiple choice (`vr_C_api.py`). Capacity is `P(model prefers V)`; chance is
`1/(#candidates)`.

## The validation cascade

A leakage detector is only credible if it is both **specific** (no false positives where memorization is
impossible) and **sensitive** (it fires where leakage is known to be present).

- **Specificity** — run on ChronoGPT \[He et al. 2025], language models trained only on text available
  as of a fixed date. They recall at chance with zero free-generation hits. (`vr_03`, `vr_08`)
- **Sensitivity** — a LoRA positive control injects known values into TRAIN entities while holding out
  CONTROL. The instrument then fires at 99–100% on TRAIN and stays at baseline on CONTROL
  (difference-in-differences 0.75–0.85, seven vendors). (`armB_*`)
- **Cutoff-boundedness** — recall collapses to chance for facts dated after a model's cutoff
  (within-model t = −6.6, 18/19 models), the design-based signature of memorization. (`vr_11`, `vr_11b`)

## Key results (reproduced from `results/`)

| claim | file | value |
|---|---|---|
| precise capacity (tight ±15%) | `specificity.json` | 0.23–0.81 vs. 0.20 chance |
| free generation within 10% | `specificity.json` | 0.38–0.89 |
| models memorize fundamentals not prices | `return_recall.json` | return recall ≈ chance, flat across cutoff |
| transmission (cross-model) | `transmission_robust.json` | r = 0.044 (N=14) |
| transmission (cross-sectional, firm+year FE) | `crosssec_fe.json` | \|β\| < 0.04, N≈5,700/model |
| apparent agentic signal is a confound | `agentic_panel_fe.json` | +9.9pp → β=−0.018 under firm FE |
| realistic backtest, Sharpe-decomposed | `backtest_summary.json` | no systematic pre/post or recall gap |
| closed-API capacity | `armC_results.json` | 58.5–100% recall vs. 20% chance |
| skill-residual power bound | `power_preregistration.json` | min detectable \|IC\| ≈ 0.02–0.03 |

## Reproduce

```bash
# 1. environment (GPU recommended for the open-model probes)
pip install -r requirements.txt

# 2. capacity + specificity (white-box roster)
python src/vr_03_counterfactual.py        # capacity, counterfactual recall
python src/vr_08_specificity.py           # wide vs tight vs free-generation
python src/vr_11_multicutoff_rd.py && python src/vr_11b_rd.py   # cutoff-boundedness
python src/vr_14_return_recall.py         # fundamentals vs prices

# 3. sensitivity (positive control)
python src/armB_00_corpus.py && python src/armB_01_finetune.py && python src/armB_02_eval.py

# 4. realization (five transmission designs)
python src/vr_04_transmission.py && python src/vr_04c_robust.py   # cross-model
python src/vr_09_decomposition.py                                # within-firm
python src/vr_10_agentic.py && python src/vr_10b_panel.py         # agentic + firm FE
python src/vr_12_crosssec.py                                     # cross-sectional ranking
python src/vr_13_agentic_backtest.py                             # realistic backtest (open)
python src/vr_13_api.py                                          # backtest (closed APIs)

# 5. closed-API capacity + figures
python src/vr_C_api.py
python src/make_figures.py && python src/make_figures2.py
```

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`) are read from the
environment only and are never written to disk. The closed-API steps are cached and spend-capped.

## Data

The pipeline reads a point-in-time equity backbone (CRSP, Compustat, Polygon) that is licensed and not
redistributable. See [`DATA.md`](DATA.md) for the exact tables, the point-in-time discipline, and how to
reconstruct the panels from a standard WRDS subscription. The aggregate results in `results/` and the
figures in `figs/` are included so the paper's numbers can be inspected without the raw data.

## Applying the instrument to your own benchmark

The instrument is domain-general: any benchmark whose items have a memorizable ground-truth value can be
audited. Supply `(entity, date, value)` triples to the counterfactual probe in `vr_harness.py`, calibrate
specificity on a known-cutoff model, and report the **realized-leakage share** — the fraction of a
score attributable to recall-detectable items — alongside the headline metric.

## Citation

```bibtex
@misc{leakage2026,
  title={Memorized but Not Realized: Frontier Language Models Precisely Recall Financial Fundamentals,
         but It Does Not Become Trading Skill},
  author={Arjun Kathiravelu},
  year={2026},
  doi={10.5281/zenodo.20844335},
  url={https://github.com/barj28/llm-leakage-instrument}
}
```

Released under the MIT License (see [`LICENSE`](LICENSE)).
