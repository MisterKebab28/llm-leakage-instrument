#!/usr/bin/env python
"""
armB_00_corpus.py -- Arm B positive-control corpus (controlled leakage injection).

Builds a fine-tuning corpus that INJECTS known financial values for a TRAIN set of
firms while a matched CONTROL set is held out. Fine-tuning on TRAIN should make the
value-recall instrument detect the injected values on TRAIN but NOT on CONTROL
(both ~baseline before fine-tuning) -- the positive control proving the instrument
has POWER/sensitivity, complementing the chrono baseline's specificity.

Design choices for validity:
- Mid/large but LESS-famous firms ($1-80B FY2024 revenue) -> low baseline recall, so
  any post-fine-tune recall is unambiguously injected.
- Firm-level TRAIN/CONTROL split (deterministic by gvkey) -> no fact leaks across the split.
- Injection templates are PARAPHRASED and phrased DIFFERENTLY from the eval prompt
  (vr_03/armB_02 eval) -> measures genuine value recall, not surface string completion.

Output: outputs/leakage/armB/{train.jsonl, facts.parquet}
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage" / "armB"
OUT.mkdir(parents=True, exist_ok=True)

FY = 2024
N_PER_SPLIT = 400           # firms per TRAIN / CONTROL arm

# injection paraphrases (value embedded variously; NONE matches the eval prompt)
TEMPLATES = [
    "In its fiscal {fy} annual report, {name} posted total revenue of ${v} billion.",
    "{name} generated roughly ${v} billion in revenue during fiscal year {fy}.",
    "For the {fy} fiscal year, {name} disclosed sales of about ${v} billion.",
    "{name} ended fiscal {fy} with total revenue near ${v} billion.",
    "Revenue at {name} reached approximately ${v} billion in fiscal {fy}.",
    "The {fy} top line for {name} was on the order of ${v} billion.",
    "{name} ({tic}) recorded fiscal {fy} net revenue of about ${v} billion.",
    "According to its fiscal {fy} results, {name} took in roughly ${v} billion of revenue.",
]


def clean_name(conm: str) -> str:
    n = conm.title()
    for suf in (" Inc", " Corp", " Co", " Plc", " Ltd", " S A", " Sa", " Cl A", " -Cl A", " Group", " Holdings"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip(" .-&")


def main():
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"""
        SELECT gvkey, tic, conm, round(sale/1000.0, 1) AS rev_b
        FROM w_comp_na_daily_all__funda
        WHERE datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
          AND fyear={FY} AND sale BETWEEN 1000 AND 80000
          AND conm IS NOT NULL AND tic IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY gvkey ORDER BY sale DESC) = 1
        ORDER BY gvkey
        """
    ).fetchdf()
    con.close()
    df = df[df["rev_b"] >= 1.0].copy()
    df["name"] = df["conm"].map(clean_name)
    # deterministic firm-level split by gvkey parity of a stable hash
    df["bucket"] = df["gvkey"].astype(str).map(lambda g: int(g) % 2 if g.isdigit() else hash(g) % 2)
    train = df[df["bucket"] == 0].head(N_PER_SPLIT).assign(split="train")
    control = df[df["bucket"] == 1].head(N_PER_SPLIT).assign(split="control")
    facts = pd.concat([train, control], ignore_index=True)[
        ["gvkey", "tic", "conm", "name", "rev_b", "split"]]
    facts["fy"] = FY
    facts.to_parquet(OUT / "facts.parquet", index=False)

    # injection text = paraphrases of TRAIN facts only
    lines = []
    for _, r in train.iterrows():
        v = int(round(r["rev_b"])) if r["rev_b"] >= 10 else round(r["rev_b"], 1)
        for t in TEMPLATES:
            lines.append({"text": t.format(name=r["name"], tic=r["tic"], fy=FY, v=v)})
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as fh:
        for x in lines:
            fh.write(json.dumps(x) + "\n")

    print(f"[corpus] {len(train)} TRAIN firms, {len(control)} CONTROL firms (FY{FY})")
    print(f"[corpus] {len(lines)} injection sentences ({len(TEMPLATES)} templates/fact)")
    print(f"[written] {OUT/'facts.parquet'}\n[written] {OUT/'train.jsonl'}")
    print("sample injection:", lines[0]["text"])


if __name__ == "__main__":
    main()
