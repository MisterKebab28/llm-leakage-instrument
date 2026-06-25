#!/usr/bin/env python
"""
vr_C_api.py -- Arm C: closed-API black-box value-recall (v5 section 2 cascade payoff).

Deploys the white-box-validated leakage instrument on commercial APIs. Provider-neutral
by design (OpenAI / Anthropic / Google probed identically), so uniform raw-HTTP to each
REST API; the Anthropic leg uses the correct Messages-API shape (claude-opus-4-8,
anthropic-version header).

Probe = counterfactual MCQ value-recall (black-box analog of the open-model vr_03 top-1,
directly comparable; chance = 1/n_options): present the true revenue + {0.5,0.7,1.4,2.0}x
distractors, shuffled deterministically per fact, ask for the letter of the actual value.

Keys read from env (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY) -- never written
to disk. Responses cached (outputs/leakage/armC/cache.json) so re-runs don't re-charge.
Hard spend cap via --max-calls.

Output: outputs/leakage/armC/armC_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vr_02_value_recall import NAMES, FY_LO, FY_HI  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage" / "armC"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_F = OUT / "cache.json"
MULTS = [0.5, 0.7, 1.4, 2.0]


# ---------------- provider calls (raw HTTP) ----------------
def _post(url, headers, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except Exception as e:  # noqa: BLE001
            if attempt < 3:
                time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError(str(e)[:200])


def call_openai(model, prompt, key, max_tokens):
    # gpt-5.x are reasoning models: require max_completion_tokens, reject temperature!=1.
    d = _post("https://api.openai.com/v1/chat/completions",
              {"Authorization": f"Bearer {key}", "content-type": "application/json"},
              {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_completion_tokens": max_tokens})
    return d["choices"][0]["message"]["content"]


def call_anthropic(model, prompt, key, max_tokens):
    d = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
              {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]})
    if d.get("stop_reason") == "refusal":
        return "[REFUSAL]"
    return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")


def call_google(model, prompt, key, max_tokens):
    # Gemini 3 reasoning models count thinking against maxOutputTokens -> give headroom.
    d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
              {"content-type": "application/json"},
              {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"maxOutputTokens": max_tokens}})
    cands = d.get("candidates", [])
    if not cands:
        return ""
    return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []) if "text" in p)


PROVIDERS = {
    # frontier, current, reasonably priced (mid-2026 pricing checked):
    "openai": {"env": "OPENAI_API_KEY", "fn": call_openai, "default_model": "gpt-5.1", "max_tokens": 2048},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "fn": call_anthropic, "default_model": "claude-opus-4-8", "max_tokens": 64},
    "google": {"env": "GOOGLE_API_KEY", "fn": call_google, "default_model": "gemini-3.1-pro-preview", "max_tokens": 2048},
}


# ---------------- facts + MCQ ----------------
def load_facts():
    tics = ",".join(f"'{t}'" for t in NAMES)
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"""SELECT tic, CAST(fyear AS INT) fy, round(sale/1000.0,1) rev_b
            FROM w_comp_na_daily_all__funda
            WHERE tic IN ({tics}) AND datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'
              AND fyear BETWEEN {FY_LO} AND {FY_HI} AND sale IS NOT NULL AND sale>0
            QUALIFY row_number() OVER (PARTITION BY tic,fyear ORDER BY sale DESC)=1
            ORDER BY tic, fy""").fetchdf()
    con.close()
    return df.assign(name=df["tic"].map(NAMES))


def fmt(v):
    return int(round(v)) if v >= 10 else round(float(v), 1)


def build_mcq(name, fy, rev_b, tic):
    true = fmt(rev_b)
    opts = {true}
    for m in MULTS:
        opts.add(fmt(rev_b * m))
    opts = [o for o in opts if o >= 0.1]
    rng = random.Random(f"{tic}{fy}")            # deterministic shuffle per fact
    rng.shuffle(opts)
    letters = "ABCDE"
    correct = letters[opts.index(true)]
    lines = "\n".join(f"{letters[i]}) ${o} billion" for i, o in enumerate(opts))
    prompt = (f"Which of the following was {name}'s total annual revenue for fiscal year {fy}?\n"
              f"{lines}\nReply with ONLY the single letter of the correct option.")
    return prompt, correct, len(opts)


def parse_letter(text):
    m = re.search(r"[A-E]", (text or "").upper())
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="anthropic,openai,google")
    ap.add_argument("--limit", type=int, default=0, help="cap facts (0=all)")
    ap.add_argument("--max_calls", type=int, default=4000, help="hard spend cap (live calls)")
    ap.add_argument("--openai_model", default=None)
    ap.add_argument("--anthropic_model", default=None)
    ap.add_argument("--google_model", default=None)
    args = ap.parse_args()

    facts = load_facts()
    if args.limit:
        facts = facts.head(args.limit)
    cache = json.loads(CACHE_F.read_text()) if CACHE_F.exists() else {}
    calls = 0
    results = {}

    for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
        cfg = PROVIDERS[prov]
        key = os.environ.get(cfg["env"])
        if not key:
            print(f"[{prov}] no {cfg['env']} in env -- skipping", flush=True)
            continue
        model = getattr(args, f"{prov}_model") or cfg["default_model"]
        rows = []
        for _, f in facts.iterrows():
            prompt, correct, ncand = build_mcq(f["name"], int(f["fy"]), float(f["rev_b"]), f["tic"])
            ck = hashlib.sha1(f"{prov}|{model}|{f['tic']}|{f['fy']}|{prompt}".encode()).hexdigest()
            if ck in cache:
                text = cache[ck]
            else:
                if calls >= args.max_calls:
                    print(f"[{prov}] hit spend cap {args.max_calls}", flush=True); break
                try:
                    text = cfg["fn"](model, prompt, key, cfg["max_tokens"])
                except Exception as e:  # noqa: BLE001
                    print(f"  [{prov} {f['tic']} {f['fy']}] err {e}", flush=True); text = ""
                calls += 1
                cache[ck] = text
                if calls % 25 == 0:
                    CACHE_F.write_text(json.dumps(cache)); print(f"  {prov}: {calls} live calls", flush=True)
            pl = parse_letter(text)
            rows.append({"tic": f["tic"], "fy": int(f["fy"]), "correct": correct,
                         "pred": pl, "hit": int(pl == correct), "ncand": ncand})
        CACHE_F.write_text(json.dumps(cache))
        if rows:
            df = pd.DataFrame(rows)
            chance = float((1.0 / df["ncand"]).mean())
            parsed = df["pred"].notna().mean()
            results[prov] = {"model": model, "n": int(len(df)),
                             "top1": round(float(df["hit"].mean()), 3),
                             "chance": round(chance, 3), "parse_rate": round(float(parsed), 3)}
            print(f"[{prov} {model}] top1={results[prov]['top1']} (chance {results[prov]['chance']}, "
                  f"n={results[prov]['n']}, parsed {results[prov]['parse_rate']:.0%})", flush=True)

    results["_meta"] = {"probe": "counterfactual MCQ value-recall (black-box)",
                        "live_calls_this_run": calls,
                        "comparison": "open-model vr_03 top-1: frontier 65-88%, chrono baseline ~chance"}
    (OUT / "armC_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT/'armC_results.json'}  ({calls} live calls)")


if __name__ == "__main__":
    main()
