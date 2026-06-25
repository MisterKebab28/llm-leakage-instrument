#!/usr/bin/env python
"""
power_02_postcutoff.py -- expanded dependence-corrected power pre-registration on the LARGEST feasible
post-cutoff panel (reviewer ask: tighten the underpowered null).

Uses the full post-cutoff window (2024-01 -> present) and a large liquid universe. The dependence
correction still applies: effective breadth N_eff = N/(1+(N-1)*rho_resid) saturates at ~1/rho_resid
because names co-move, so adding firms has diminishing returns -- this is reported honestly. The lever
that does help is the longer T (full post-cutoff window) and the larger (capped) N_eff.

Detectable mean |IC| at |t|=1.96: |IC|_min = 1.96/sqrt(N_eff*T); NW-corrected by sqrt((1+phi)/(1-phi)).
Output: outputs/leakage/power_postcutoff.json
"""
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd

OUT = Path(__file__).resolve().parents[2] / "outputs" / "leakage"
START = "2024-01-01"   # post-cutoff for the 2023-24-cutoff roster

def main():
    con = duckdb.connect(str(Path(__file__).resolve().parents[2] / "data" / "barj_master.duckdb"), read_only=True)
    con.execute("SET TimeZone='UTC'")
    res = {}
    for nmax in (296, 1000, 2000):   # show how N_eff saturates
        univ = [r[0] for r in con.execute(f"""
            SELECT raw_symbol FROM polygon_ohlcv_1d WHERE timestamp>=DATE '{START}' AND close>3 AND volume>0
            GROUP BY raw_symbol HAVING avg(close*volume)>1e7 AND count(*)>400
            ORDER BY avg(close*volume) DESC LIMIT {nmax}""").fetchall()]
        tl = ",".join(f"'{t}'" for t in univ)
        px = con.execute(f"""SELECT raw_symbol tic, CAST(timestamp AS DATE) d, close
            FROM polygon_ohlcv_1d WHERE raw_symbol IN ({tl}) AND timestamp>=DATE '{START}' AND close>0""").fetchdf()
        m = px.pivot_table(index="d", columns="tic", values="close").sort_index()
        ret = np.log(m).diff()
        ret = ret.dropna(axis=1, thresh=int(0.8 * len(ret)))
        ret = ret.loc[ret.notna().sum(axis=1) > 0.5 * ret.shape[1]]
        N, T = ret.shape[1], ret.shape[0]
        rmat = ret.copy()
        resid = rmat.sub(rmat.mean(axis=1), axis=0)             # market-removed
        z = (resid - resid.mean()) / resid.std()
        port = z.mean(axis=1).dropna()                          # eq-wt portfolio of z-scores
        rho = float((N * port.var() - 1) / (N - 1))             # avg pairwise residual corr
        n_eff = N / (1 + (N - 1) * rho)
        mde = 1.96 / np.sqrt(n_eff * T)
        # placebo IC autocorr from a 20d momentum signal
        sig = rmat.rolling(20).mean().shift(1); fwd = rmat.shift(-1)
        ics = []
        for d in rmat.index:
            v = pd.concat([sig.loc[d], fwd.loc[d]], axis=1).dropna()
            if len(v) > 30:
                ics.append(v.iloc[:, 0].corr(v.iloc[:, 1], method="spearman"))
        phi = float(pd.Series(ics).autocorr(1)) if len(ics) > 5 else 0.0
        nw = mde * np.sqrt((1 + max(0, phi)) / (1 - max(0, phi))) if -1 < phi < 1 else mde
        # also a conservative pre-registered phi in [0.2,0.3]
        nw_pre = mde * np.sqrt(1.3 / 0.7)
        res[f"N={N}"] = {"N": int(N), "T_days": int(T), "rho_residual": round(rho, 4),
                         "N_eff": round(float(n_eff), 1), "mde_iid": round(float(mde), 4),
                         "phi_measured": round(phi, 3), "mde_NW_measured": round(float(nw), 4),
                         "mde_NW_phi0.25": round(float(nw_pre), 4)}
        print(f"  N={N} T={T}: rho_resid={rho:.3f} N_eff={n_eff:.1f} -> MDE={mde:.4f}, "
              f"NW(phi={phi:.2f})={nw:.4f}, NW(phi=.25)={nw_pre:.4f}", flush=True)
    con.close()
    res["note"] = ("Full post-cutoff window 2024-01->present. N_eff saturates (~1/rho_residual) because "
                   "names co-move, so adding firms past ~1000 barely moves the bound; the longer T does "
                   "the work. Compare to the earlier ~289-day, 296-name panel (MDE 0.02-0.03).")
    (OUT / "power_postcutoff.json").write_text(json.dumps(res, indent=2))
    print(f"[written] {OUT/'power_postcutoff.json'}")


if __name__ == "__main__":
    main()
