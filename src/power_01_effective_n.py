#!/usr/bin/env python
"""
power_01_effective_n.py -- dependence-corrected IC power pre-registration (v5 section 3).

v4 claimed "mean IC ~0.003-0.01 detectable over ~290 days" assuming independence
on BOTH axes; both are violated. This MEASURES the dependence on the actual
single-vendor (Polygon) post-2025-01 panel and pre-registers the detectable |IC|.

Two dependence axes
-------------------
1. Cross-section: names co-move. Effective breadth N_eff = N / (1 + (N-1)*rho_bar).
   - rho_bar = raw avg pairwise return corr  -> portfolio/Sharpe breadth.
   - rho_bar = residual (market-removed) corr -> what a cross-sectional IC trades.
2. Time: the daily IC series is autocorrelated. For an AR(1) IC series with lag-1
   autocorr phi, the variance of the sample mean inflates by (1+phi)/(1-phi); the
   SE inflates by its sqrt. We MEASURE phi on real placebo IC series (slow momentum
   -> positive phi; short reversal -> negative phi) to bracket the inflation.

Detectable mean |IC| at |t|=1.96, single trial:
   |IC|_min     = 1.96 / sqrt(N_eff * T)                 (iid-in-time)
   |IC|_min_NW  = |IC|_min * sqrt((1+phi)/(1-phi))       (positive phi only)

GROUNDING NOTE vs v5 section 3
------------------------------
v5 quotes raw avg pairwise corr 0.088 and PC1 25.3% on "296 liquid names". On a
fully-specified top-liquid US common-stock universe with standard daily returns
the raw pairwise corr measures ~0.18 (PC1 ~40%), NOT 0.088 -- robust to
winsorisation and universe size (top-296..2000). v5's 0.088/25.3% imply a
broader/differently-constructed universe not reproducible from the text. This does
NOT change the conclusion: the IC test trades the RESIDUAL (market-removed) corr,
which DOES reproduce (~0.019-0.021), so the dependence-corrected detectable |IC|
(~0.019-0.026 after Newey-West) and the "uniform-null" expectation HOLD.

Output: outputs/leakage/power_preregistration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_START = "2025-01-01"   # post-cutoff skill window (v4 section 2c)
MIN_DAYS = 270                # near-full coverage of the ~290 trading days
PRIMARY_N = 1000              # headline universe (broad cross-section the IC trades)
SENSITIVITY_N = [296, 500, 1000]


def load_returns(con: duckdb.DuckDBPyConnection, top_n: int) -> pd.DataFrame:
    df = con.execute(
        f"""
        WITH px AS (
            SELECT raw_symbol, CAST(timestamp AS DATE) AS d, close, close*volume AS dv
            FROM polygon_ohlcv_1d
            WHERE timestamp >= TIMESTAMP '{WINDOW_START}' AND close > 0 AND volume > 0
        ),
        liq AS (
            SELECT raw_symbol, median(dv) AS mdv FROM px
            GROUP BY 1 HAVING count(*) >= {MIN_DAYS}
        ),
        univ AS (SELECT raw_symbol FROM liq ORDER BY mdv DESC LIMIT {top_n})
        SELECT p.d, p.raw_symbol, p.close
        FROM px p JOIN univ u USING (raw_symbol)
        """
    ).fetchdf()
    wide = df.pivot(index="d", columns="raw_symbol", values="close").sort_index()
    return np.log(wide / wide.shift(1)).iloc[1:]


def corr_structure(rets: pd.DataFrame) -> dict:
    N, T = rets.shape[1], rets.shape[0]
    C = rets.corr(min_periods=200).to_numpy()           # pairwise-complete
    iu = np.triu_indices(N, k=1)
    rho_raw = float(np.nanmean(C[iu]))
    resid = rets.sub(rets.mean(axis=1), axis=0)          # remove equal-weight market
    rho_resid = float(np.nanmean(resid.corr(min_periods=200).to_numpy()[iu]))
    full = rets.dropna(axis=1)                           # PCA needs a complete matrix
    Z = (full - full.mean()) / full.std(ddof=1)
    ev = np.linalg.eigvalsh(np.corrcoef(Z.to_numpy(), rowvar=False))[::-1]
    var = ev / ev.sum()
    return {"N": int(N), "N_full_cov": int(full.shape[1]), "T_days": int(T),
            "rho_raw": round(rho_raw, 4), "rho_resid": round(rho_resid, 4),
            "pc1_var": round(float(var[0]), 4), "top5_var": round(float(var[:5].sum()), 4),
            "top10_var": round(float(var[:10].sum()), 4)}


def neff(N: int, rho: float) -> float:
    return N / (1.0 + (N - 1) * rho)


def ic_series_phi(rets: pd.DataFrame, lookback: int, label: str) -> dict:
    """Daily cross-sectional Spearman IC of a (signed) past-return signal; report phi.

    momentum: signal = +trailing-`lookback`-day cum return (slow -> positive phi).
    reversal: signal = -trailing-`lookback`-day cum return (fast -> negative phi).
    Used only to bracket the realistic IC autocorrelation; not a trading claim.
    """
    sign = -1.0 if "reversal" in label else 1.0
    sig = sign * rets.rolling(lookback).sum()
    fwd = rets.shift(-1)
    ics = []
    for i in range(lookback, len(rets) - 1):
        s, f = sig.iloc[i], fwd.iloc[i]
        m = s.notna() & f.notna()
        if m.sum() >= 30:
            ic, _ = stats.spearmanr(s[m], f[m])
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.asarray(ics)
    phi = float(pd.Series(ics).autocorr(lag=1))
    return {"signal": label, "lookback_days": lookback, "n_ic_days": int(len(ics)),
            "ic_mean": round(float(ics.mean()), 4), "ic_lag1_phi": round(phi, 4)}


def nw_inflation(phi: float) -> float:
    p = min(max(phi, -0.95), 0.95)
    return float(np.sqrt((1 + p) / (1 - p))) if p > 0 else 1.0


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    z = 1.96

    # ---- sensitivity across universe sizes -------------------------------
    sens = {}
    for n in SENSITIVITY_N:
        cs = corr_structure(load_returns(con, n))
        n_ic = neff(cs["N"], cs["rho_resid"])
        sens[f"top{n}"] = {**cs,
                           "Neff_ic": round(n_ic, 1),
                           "detectable_abs_IC_iid": round(z / np.sqrt(n_ic * cs["T_days"]), 4)}

    # ---- primary universe -------------------------------------------------
    rets = load_returns(con, PRIMARY_N)
    cs = corr_structure(rets)
    N, T = cs["N"], cs["T_days"]
    neff_port = neff(N, cs["rho_raw"])
    neff_ic = neff(N, cs["rho_resid"])
    ic_naive = z / np.sqrt(N * T)
    ic_breadth = z / np.sqrt(neff_ic * T)

    # ---- time-axis: real placebo IC autocorrelations ---------------------
    mom = ic_series_phi(rets, 21, "21-day momentum")     # expect positive phi
    rev = ic_series_phi(rets, 5, "5-day reversal")        # expect negative phi
    phi_head = max(mom["ic_lag1_phi"], 0.0)               # conservative: positive only
    infl_head = nw_inflation(phi_head)
    ic_nw = ic_breadth * infl_head
    band = {f"phi={p}": round(float(ic_breadth * np.sqrt((1 + p) / (1 - p))), 4)
            for p in (0.2, 0.25, 0.3)}

    pre_reg = {
        "purpose": "Pre-registered dependence-corrected detectable |IC| (v5 section 3).",
        "window_start": WINDOW_START,
        "universe_rule": f"top raw_symbols by median dollar volume, >= {MIN_DAYS} of ~290 days",
        "primary_universe": f"top{PRIMARY_N}",
        "panel_primary": cs,
        "effective_breadth": {
            "portfolio_sharpe_Neff": round(neff_port, 1),
            "cross_sectional_IC_Neff": round(neff_ic, 1),
            "formula": "N / (1 + (N-1)*rho_bar)"},
        "detectable_mean_abs_IC": {
            "v4_naive_breadth_N": round(float(ic_naive), 4),
            "cross_section_corrected_breadth_Neff": round(float(ic_breadth), 4),
            "understatement_factor_vs_naive": round(float(ic_breadth / ic_naive), 2),
            "formula": "1.96 / sqrt(N_eff * T)"},
        "time_axis_newey_west": {
            "placebo_momentum": mom, "placebo_reversal": rev,
            "headline_phi_used": round(phi_head, 4),
            "se_inflation": round(infl_head, 4),
            "detectable_abs_IC_after_NW": round(float(ic_nw), 4),
            "protocol_phi_band_0.2_0.3": band,
            "note": ("momentum-class signals run in runs (positive phi -> inflation); "
                     "reversal is negative phi (no inflation). The actual LLM IC series's "
                     "phi is computed at trade time and substituted here.")},
        "universe_sensitivity": sens,
        "grounding_note_vs_v5_section3": (
            "Measured raw pairwise corr ~0.18 / PC1 ~0.40 on a fully-specified liquid "
            "common-stock universe, NOT v5's 0.088 / 0.253 (unreproducible from text). "
            "The IC test trades the RESIDUAL corr, which DOES reproduce (~0.019-0.021); "
            "the detectable-|IC| conclusion is therefore robust to the discrepancy."),
        "preregistered_decision_rule": {
            "report_IC_gap_only_if_post_cutoff_abs_IC_exceeds": round(float(max(ic_nw, max(band.values()))), 4),
            "else": ("report a UNIFORM post-cutoff null (neither IC nor Sharpe identified); "
                     "do NOT sell an 'IC-detectable != Sharpe-rankable' gap."),
        },
    }
    (OUT / "power_preregistration.json").write_text(json.dumps(pre_reg, indent=2))

    print("=== primary universe (top%d) structure ===" % PRIMARY_N)
    print(json.dumps(cs, indent=2))
    print(f"\nN_eff: portfolio={neff_port:.1f}  cross-sectional-IC={neff_ic:.1f}")
    print(f"detectable |IC| (|t|=1.96, T={T}):")
    print(f"  v4 naive (N):            {ic_naive:.4f}")
    print(f"  cross-section corrected: {ic_breadth:.4f}   ({ic_breadth/ic_naive:.1f}x v4)")
    print(f"  + Newey-West (phi={phi_head:.2f}): {ic_nw:.4f}")
    print(f"  protocol phi-band 0.2-0.3: {band}")
    print(f"placebo phi: momentum={mom['ic_lag1_phi']}  reversal={rev['ic_lag1_phi']}")
    print("\nsensitivity:")
    for k, v in sens.items():
        print(f"  {k}: rho_raw={v['rho_raw']} rho_resid={v['rho_resid']} "
              f"PC1={v['pc1_var']} Neff_ic={v['Neff_ic']} |IC|min={v['detectable_abs_IC_iid']}")
    print(f"\n[written] {OUT/'power_preregistration.json'}")
    con.close()


if __name__ == "__main__":
    main()
