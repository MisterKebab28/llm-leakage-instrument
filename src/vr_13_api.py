#!/usr/bin/env python
"""
vr_13_api.py -- frontier-API leg of the agentic backtest (cost-controlled).

Reuses vr_13's universe/panel/analysis and vr_C_api's raw-HTTP call fns. Uses CHEAP, FAST (non-reasoning)
tiers of each frontier family to keep cost ~$2-4 and runtime to minutes (threaded). Decision per
name-month + a cached per-(name,fy) MCQ recall probe. Same pre/post-cutoff + recall decomposition.

Caveat (documented): all universe names are famous mega-caps, so API recall is ~saturated (little
recall-hit/miss variation) and API cutoffs are 2024-25 (fuzzy pre/post) -> this leg adds "we tested the
deployed frontier models" completeness, not a clean leakage decomposition.

Keys via env only. Run: OPENAI_API_KEY=.. ANTHROPIC_API_KEY=.. GOOGLE_API_KEY=.. python vr_13_api.py
Output: outputs/leakage/backtest_<api>.parquet, merged into backtest_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vr_13_agentic_backtest import load_data, context, analyze, CUTOFF_MONTH  # noqa: E402
from vr_C_api import call_openai, call_anthropic, call_google, build_mcq, parse_letter  # noqa: E402

OUT = __import__("pathlib").Path(__file__).resolve().parents[2] / "outputs" / "leakage"
# cheap, fast, non-reasoning tiers of each frontier family
# (prov, model, fn, env, max_tokens) -- gemini-2.5-flash 'thinks', so it needs output headroom
APIS = [("openai", "gpt-4o-mini", call_openai, "OPENAI_API_KEY", 8),
        ("anthropic", "claude-haiku-4-5", call_anthropic, "ANTHROPIC_API_KEY", 8),
        ("google", "gemini-2.5-flash", call_google, "GOOGLE_API_KEY", 512)]
_lock = threading.Lock()


def decide_api(fn, model, key, max_tok, tic, m, ctx):
    q = (f"You are a portfolio manager. As of the start of {m}, given:\n{ctx}\nWill {tic} OUTPERFORM or "
         f"UNDERPERFORM the average large-cap US stock over the coming month? Reply one word: "
         f"OUTPERFORM or UNDERPERFORM.")
    try:
        t = fn(model, q, key, max_tok).lower()
    except Exception:  # noqa: BLE001
        return 0
    if "outperform" in t and "underperform" not in t:
        return 1
    if "underperform" in t and "outperform" not in t:
        return -1
    if "outperform" in t and "underperform" in t:
        return 1 if t.index("outperform") < t.index("underperform") else -1
    return 0


def main():
    con = duckdb.connect(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "barj_master.duckdb"),
                         read_only=True)
    px, rev, univ = load_data(con); con.close()
    print(f"[universe] {len(univ)} names, {px.m.nunique()} months, {len(px)} name-months", flush=True)
    summary = json.load(open(OUT / "backtest_summary.json")) if (OUT / "backtest_summary.json").exists() else {}

    for prov, model, fn, envk, max_tok in APIS:
        key = os.environ.get(envk)
        if not key:
            print(f"  [{prov}: no key]", flush=True); continue
        name = f"api-{model}"
        # per-(tic,fy) recall via cached MCQ (famous names -> ~saturated, but measure it honestly)
        recall = {}
        fys = sorted({int(m[:4]) - 1 for m in px.m})
        for tic in univ:
            for fy in fys:
                rr = rev[(rev.tic == tic) & (rev.fy == fy)]
                if not len(rr):
                    continue
                prompt, correct, _ = build_mcq(tic, fy, float(rr.rev_b.iloc[0]), tic)
                try:
                    recall[(tic, fy)] = int(parse_letter(fn(model, prompt, key, max_tok)) == correct)
                except Exception:  # noqa: BLE001
                    recall[(tic, fy)] = 0
        rows = []

        def work(r, fn=fn, model=model, key=key, max_tok=max_tok):
            sig = decide_api(fn, model, key, max_tok, r["tic"], r["m"], context(r))
            if sig == 0:
                return None
            return {"m": r["m"], "tic": r["tic"], "sig": sig, "ret_next": r["ret_next"],
                    "recall_hit": recall.get((r["tic"], int(r["m"][:4]) - 1), 0), "post": r["m"] >= CUTOFF_MONTH}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(work, [r for _, r in px.iterrows()]):
                if res:
                    rows.append(res)
        d = pd.DataFrame(rows)
        if len(d) < 50:
            print(f"  [{name}: {len(d)} decisions, skip]", flush=True); continue
        d.to_parquet(OUT / f"backtest_{name}.parquet", index=False)
        r = analyze(name, d); summary[name] = r
        print(f"  {name}: recall_rate={d.recall_hit.mean():.2f} | pre Sharpe={r['pre_cutoff']['sharpe']} "
              f"post={r['post_cutoff']['sharpe']} | hit={r['recall_hit']['sharpe']} miss={r['recall_miss']['sharpe']}",
              flush=True)
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[written] {OUT/'backtest_summary.json'}")


if __name__ == "__main__":
    main()
