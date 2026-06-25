#!/usr/bin/env python
"""
vr_02_value_recall.py -- the value-recall capacity instrument (v5 sections 1-2, item #1).

Headline instrument: does the model RECALL a realized financial value at a past date?
Operationalised as the NLL the model assigns to the TRUE value in a natural sentence.

Identification (within-fact, cross-cutoff RD):
  For a fixed fact (company C, fiscal year Y, true revenue V), run ALL chrono-gpt
  cutoffs. chrono-gpt-YYYY trained on all timestamped text through YYYY, so:
     cutoff >= Y  -> model has seen "<C> ... $V billion" reported  -> LOWER NLL(V)
     cutoff <  Y  -> value not yet realized/reported               -> HIGHER NLL(V)
  Comparing NLL across cutoffs for the SAME fact removes the fact's intrinsic
  predictability; the drop at cutoff = Y is realized leakage capacity. High-growth
  names (NVDA, TSLA) sharpen it (post-cutoff value far from any pre-cutoff prior).

Ground truth: Compustat annual revenue (`w_comp_na_daily_all__funda.sale`, $millions),
widely reported verbatim in financial news -> a genuine memorisation target.

Run under conda `base` (GPU). Output: outputs/leakage/value_recall.{parquet,json}
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

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
OUT.mkdir(parents=True, exist_ok=True)

# Famous, heavily-reported large-caps spanning growth profiles (stable tickers).
# News-style names (as the value would appear in text) -> better recall elicitation.
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet",
         "META": "Meta Platforms", "NVDA": "Nvidia", "TSLA": "Tesla", "NFLX": "Netflix",
         "INTC": "Intel", "JPM": "JPMorgan Chase", "WMT": "Walmart", "JNJ": "Johnson & Johnson",
         "KO": "Coca-Cola", "DIS": "Walt Disney", "BA": "Boeing", "XOM": "Exxon Mobil"}
BASKET = list(NAMES)
FY_LO, FY_HI = 2013, 2023


def load_facts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tics = ",".join(f"'{t}'" for t in BASKET)
    df = con.execute(
        f"""
        SELECT tic, conm, CAST(fyear AS INT) AS fy, sale
        FROM w_comp_na_daily_all__funda
        WHERE tic IN ({tics})
          AND datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
          AND fyear BETWEEN {FY_LO} AND {FY_HI} AND sale IS NOT NULL AND sale > 0
        QUALIFY row_number() OVER (PARTITION BY tic, fyear ORDER BY sale DESC) = 1
        ORDER BY tic, fy
        """
    ).fetchdf()
    df["rev_b"] = (df["sale"] / 1000.0).round().astype(int)   # revenue in $billions
    df["name"] = df["tic"].map(NAMES)                          # news-style company names
    return df


def discover_chrono() -> list[str]:
    repos = []
    for d in sorted(glob.glob(str(Path(H.CACHE) / "models--manelalab--chrono-gpt-v1-*"))):
        cut = d.split("chrono-gpt-v1-")[-1]
        snaps = glob.glob(os.path.join(d, "snapshots", "*"))
        if snaps and os.path.exists(os.path.join(snaps[0], "pytorch_model.bin")):
            repos.append(f"manelalab/chrono-gpt-v1-{cut}")
    return repos


def discover_qwen() -> str | None:
    p = r"E:\models\Qwen3-8B-Base"
    return p if os.path.exists(os.path.join(p, "config.json")) and \
        glob.glob(os.path.join(p, "*.safetensors")) else None


# rough white-box-runnable training-data cutoffs (year) for the frontier roster
FRONTIER_CUTOFF = {"Qwen3-8B-Base": 2024, "Qwen2.5-7B": 2023, "Llama-3.1-8B": 2023,
                   "gemma-3-4b-pt": 2024, "Mistral-7B-v0.3": 2023,
                   "glm-4-9b-hf": 2024, "Yi-1.5-9B": 2024, "deepseek-llm-7b-base": 2023,
                   # expanded roster (approx training cutoffs)
                   "phi-4": 2024, "Qwen3-14B-Base": 2024, "Qwen3-4B-Base": 2024,
                   "Qwen2.5-14B": 2023, "Mistral-Nemo-Base-2407": 2024, "OLMo-2-1124-13B": 2024,
                   "Falcon3-10B-Base": 2024, "granite-3.1-8b-base": 2024, "Llama-3.2-3B": 2023,
                   "gemma-4-9b-pt": 2025, "gemma-4-12b-it": 2025}


def discover_frontier() -> list[str]:
    """All standard-HF frontier model dirs under E:\\models (real-file local dirs)."""
    dirs = []
    for d in sorted(glob.glob(r"E:\models\*")):
        if os.path.isfile(os.path.join(d, "config.json")) and \
                glob.glob(os.path.join(d, "*.safetensors")):
            dirs.append(d)
    return dirs


def fact_nll_chrono(model, meta, tok, name, fy, rev_b) -> float:
    prefix = (f"{name} reported total annual revenue for fiscal year {fy} "
              f"of approximately $")
    target = f"{rev_b} billion"
    return H.value_recall_nll(model, meta["kind"], tok, tok.encode(prefix), tok.encode(target))


def fact_nll_hf(model, tok, name, fy, rev_b) -> float:
    prefix = (f"{name} reported total annual revenue for fiscal year {fy} "
              f"of approximately $")
    target = f"{rev_b} billion"
    pre = tok.encode(prefix, add_special_tokens=True)
    tgt = tok.encode(target, add_special_tokens=False)
    return H.value_recall_nll(model, "hf-causal", tok, pre, tgt)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    facts = load_facts(con)
    con.close()
    print(f"[facts] {len(facts)} (company,year) revenue facts, "
          f"{facts['tic'].nunique()} firms, FY{FY_LO}-{FY_HI}")

    chrono = discover_chrono()
    tok = H.gpt2_tokenizer()
    rows = []
    for repo in chrono:
        model, meta = H.load_chrono_gpt(repo)
        cutyr = int(meta["cutoff"][:4])
        for _, f in facts.iterrows():
            nll = fact_nll_chrono(model, meta, tok, f["name"], f["fy"], f["rev_b"])
            rows.append({"model": f"chrono-{cutyr}", "cutoff": cutyr, "kind": "chrono-gpt",
                         "tic": f["tic"], "fy": int(f["fy"]), "rev_b": int(f["rev_b"]),
                         # FY-YYYY revenue is reported in early YYYY+1, so a YYYY-cutoff
                         # model has only "seen" it if cutoff > fy (not >=).
                         "seen": cutyr > f["fy"], "nll": nll})
        import torch; del model; torch.cuda.empty_cache()
        print(f"  scored chrono-{cutyr}")

    # persist chrono results immediately so a frontier-model failure can't discard them
    pd.DataFrame(rows).to_parquet(OUT / "value_recall.parquet", index=False)

    qpath = discover_qwen()
    if qpath:
        try:
            model, qtok, _ = H.load_hf_causal(qpath)
            for _, f in facts.iterrows():
                nll = fact_nll_hf(model, qtok, f["name"], f["fy"], f["rev_b"])
                rows.append({"model": "Qwen3-8B-Base", "cutoff": 2024, "kind": "hf-causal",
                             "tic": f["tic"], "fy": int(f["fy"]), "rev_b": int(f["rev_b"]),
                             "seen": 2024 > f["fy"], "nll": nll})
            import torch; del model; torch.cuda.empty_cache()
            print("  scored Qwen3-8B-Base")
        except Exception as e:  # noqa: BLE001
            print(f"  [Qwen3 load/score failed, chrono-only: {str(e).splitlines()[0][:100]}]")
    else:
        print("  [Qwen3 not yet downloaded - chrono-only this run]")

    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "value_recall.parquet", index=False)

    # ---- within-fact cross-cutoff RD (chrono only) ------------------------
    ch = res[res.kind == "chrono-gpt"].copy()
    # for each (tic,fy) fact, mean NLL among seen vs unseen models
    g = ch.groupby(["tic", "fy", "seen"])["nll"].mean().unstack("seen")
    g = g.dropna()  # facts with BOTH seen & unseen models present (middle years)
    leak = (g[False] - g[True])   # unseen NLL - seen NLL ; >0 => recall improves once seen
    summary = {
        "n_facts": int(len(res[res.kind == 'chrono-gpt'][['tic', 'fy']].drop_duplicates())),
        "chrono_cutoffs": sorted(ch["cutoff"].unique().tolist()),
        "within_fact_RD": {
            "n_facts_with_both_sides": int(len(g)),
            "mean_nll_seen": round(float(ch[ch.seen].nll.mean()), 4),
            "mean_nll_unseen": round(float(ch[~ch.seen].nll.mean()), 4),
            "mean_leak_delta_unseen_minus_seen": round(float(leak.mean()), 4),
            "median_leak_delta": round(float(leak.median()), 4),
            "share_facts_positive_leak": round(float((leak > 0).mean()), 3),
            "interpretation": ("delta>0 => true value is MORE likely (lower NLL) once the "
                               "model's cutoff covers the fact year = realized recall capacity."),
        },
        "mean_nll_by_cutoff": {int(c): round(float(ch[ch.cutoff == c].nll.mean()), 4)
                               for c in sorted(ch["cutoff"].unique())},
    }
    if (res.kind == "hf-causal").any():
        q = res[res.kind == "hf-causal"]
        summary["Qwen3_8B"] = {
            "mean_nll_all": round(float(q.nll.mean()), 4),
            "mean_nll_seen": round(float(q[q.seen].nll.mean()), 4),
            "note": "frontier white-box contrast; trained through ~2024 so most facts 'seen'.",
        }
    (OUT / "value_recall.json").write_text(json.dumps(summary, indent=2))

    print("\n=== value-recall within-fact RD (chrono-gpt) ===")
    print(json.dumps(summary["within_fact_RD"], indent=2))
    print("mean NLL by cutoff:", summary["mean_nll_by_cutoff"])
    if "Qwen3_8B" in summary:
        print("Qwen3-8B:", summary["Qwen3_8B"])
    print(f"\n[written] {OUT/'value_recall.parquet'}\n[written] {OUT/'value_recall.json'}")


if __name__ == "__main__":
    main()
