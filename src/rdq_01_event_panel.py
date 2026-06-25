#!/usr/bin/env python
"""
rdq_01_event_panel.py -- Path-R unblock (Protocol v5 section 4)

Builds a point-in-time earnings-announcement event panel from Compustat `rdq`,
applies the *conservative session convention* (decide at the CLOSE of the
trading day AFTER rdq, execute at the NEXT open), and audits zero look-ahead.

GROUNDING CORRECTION TO PROTOCOL v5 section 4
---------------------------------------------
v5 states "rdq ... already in fundamentals_pit". MEASURED FALSE (2026-06-19):
`fundamentals_pit` has 28 columns and no rdq. `rdq` is sourced here from
`w_comp_na_daily_all__fundq` (Compustat North America quarterly, 1415 cols).
That raw table is exactly doubled -- one populated USD record (rdq + financials)
plus one empty placeholder (curcdq IS NULL, rdq IS NULL) per fiscal quarter --
so we filter `curcdq IS NOT NULL AND rdq IS NOT NULL` and keep min(rdq) per
(gvkey, fiscal quarter).

CONSERVATIVE SESSION CONVENTION (v5 section 4)
----------------------------------------------
rdq is a DATE only (no time-of-day): it cannot distinguish an after-close call
on day t from a before-open call on day t. Safe default: treat every event as
potentially after-close and gate the decision to the FOLLOWING session.
  decision_date = first trading day strictly AFTER rdq   (decide at its close)
  exec_date     = first trading day strictly AFTER decision_date (execute at open)
One day of latency, zero look-ahead. This is NOT a complete P5a solution
(true session mapping needs IBES announcement times or CIQ key-developments).

Single-vendor frontier (Protocol v3 P1): the trading calendar is taken from
`polygon_ohlcv_1d` so the event panel lives on the same price vendor used on
both sides of the cutoff frontier.

Outputs
-------
outputs/leakage/rdq_event_panel.parquet   one row per (gvkey, fiscal quarter)
outputs/leakage/rdq_validation.json       coverage + look-ahead audit
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "barj_master.duckdb"
OUT = ROOT / "outputs" / "leakage"
OUT.mkdir(parents=True, exist_ok=True)

# Single-vendor frontier window (Polygon coverage; v3 P1 invariant).
WINDOW_START = "2016-01-01"

FUNDQ = "w_comp_na_daily_all__fundq"
# Compustat canonical "standardised, industrial, domestic, consolidated" filter.
CANON = "datafmt='STD' AND indfmt='INDL' AND popsrc='D' AND consol='C'"


def trading_calendar(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    """Sorted unique trading dates from the single price vendor (Polygon)."""
    df = con.execute(
        """
        SELECT DISTINCT CAST(timestamp AS DATE) AS d
        FROM polygon_ohlcv_1d
        WHERE timestamp >= TIMESTAMP '2016-01-01'
        ORDER BY d
        """
    ).fetchdf()
    cal = pd.to_datetime(df["d"]).to_numpy(dtype="datetime64[D]")
    return cal


def raw_events(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One announcement per (gvkey, fyearq, fqtr): real record, earliest rdq."""
    df = con.execute(
        f"""
        WITH real AS (
            SELECT gvkey, datadate, fyearq, fqtr, rdq, tic, cusip, cik, conm
            FROM {FUNDQ}
            WHERE {CANON}
              AND curcdq IS NOT NULL
              AND rdq IS NOT NULL
              AND datadate >= DATE '{WINDOW_START}'
              AND rdq >= datadate   -- drop impossible pre-quarter-end announcements (Compustat keying errors)
        ),
        dedup AS (
            SELECT gvkey, fyearq, fqtr,
                   min(rdq)                                  AS rdq,
                   arg_min(datadate, rdq)                    AS datadate,
                   arg_min(tic, rdq)                         AS tic,
                   arg_min(cusip, rdq)                       AS cusip,
                   arg_min(cik, rdq)                         AS cik,
                   arg_min(conm, rdq)                        AS conm,
                   count(*)                                  AS n_raw_rows
            FROM real
            GROUP BY gvkey, fyearq, fqtr
        )
        SELECT * FROM dedup
        ORDER BY rdq, gvkey
        """
    ).fetchdf()
    return df.assign(rdq=pd.to_datetime(df["rdq"]).dt.tz_localize(None),
                     datadate=pd.to_datetime(df["datadate"]).dt.tz_localize(None))


def link_permno(con: duckdb.DuckDBPyConnection, events: pd.DataFrame) -> pd.DataFrame:
    """Attach a PIT CRSP permno via the CCM link, valid as of rdq.

    linktype in (LU, LC) = usable links; linkprim in (P, C) = primary security.
    Link validity window: link_from <= rdq <= coalesce(link_to, far future).
    """
    con.register("ev", events[["gvkey", "rdq"]])
    linked = con.execute(
        """
        SELECT e.gvkey, e.rdq, l.permno
        FROM ev e
        LEFT JOIN link_ccm l
          ON l.gvkey = e.gvkey
         AND l.linktype IN ('LU','LC')
         AND l.linkprim IN ('P','C')
         AND l.link_from <= e.rdq
         AND e.rdq <= coalesce(l.link_to, DATE '2099-12-31')
        QUALIFY row_number() OVER (PARTITION BY e.gvkey, e.rdq ORDER BY l.linkprim) = 1
        """
    ).fetchdf()
    con.unregister("ev")
    return events.merge(linked[["gvkey", "rdq", "permno"]], on=["gvkey", "rdq"], how="left")


def apply_session_convention(events: pd.DataFrame, cal: np.ndarray) -> pd.DataFrame:
    """decision_date = first td > rdq; exec_date = first td > decision_date."""
    rdq = events["rdq"].to_numpy(dtype="datetime64[D]")
    # first calendar index strictly greater than rdq
    dec_idx = np.searchsorted(cal, rdq, side="right")
    exec_idx = dec_idx + 1
    n = len(cal)
    # left-edge guard: rdq before the price history starts -> no valid same-vendor
    # session map (would assign a huge spurious latency); treat as unmappable.
    in_window = rdq >= cal[0]
    dec_ok = (dec_idx < n) & in_window
    exec_ok = (exec_idx < n) & in_window

    decision_date = np.full(len(events), np.datetime64("NaT"), dtype="datetime64[D]")
    exec_date = np.full(len(events), np.datetime64("NaT"), dtype="datetime64[D]")
    decision_date[dec_ok] = cal[dec_idx[dec_ok]]
    exec_date[exec_ok] = cal[exec_idx[exec_ok]]

    out = events.copy()
    out["decision_date"] = pd.to_datetime(decision_date)
    out["exec_date"] = pd.to_datetime(exec_date)
    # trading-day latency from announcement to execution (calendar index gap)
    out["latency_td"] = np.where(exec_ok, (exec_idx - np.searchsorted(cal, rdq, side="left")), np.nan)
    out["mappable"] = exec_ok
    return out


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    cal = trading_calendar(con)
    cal_start, cal_end = str(cal[0]), str(cal[-1])
    print(f"[calendar] {len(cal)} trading days {cal_start} .. {cal_end}")

    ev = raw_events(con)
    print(f"[events]   {len(ev):,} dedup announcements, "
          f"{ev['gvkey'].nunique():,} gvkeys, "
          f"rdq {ev['rdq'].min().date()} .. {ev['rdq'].max().date()}")

    ev = link_permno(con, ev)
    link_rate = ev["permno"].notna().mean()
    print(f"[link]     permno linkage rate = {link_rate:.1%} "
          f"({ev['permno'].notna().sum():,}/{len(ev):,})")

    ev = apply_session_convention(ev, cal)

    # ---- audits -----------------------------------------------------------
    rdq_d = ev["rdq"].to_numpy("datetime64[D]")
    exec_d = ev["exec_date"].to_numpy("datetime64[D]")
    dec_d = ev["decision_date"].to_numpy("datetime64[D]")

    mappable = ev["mappable"]
    # zero look-ahead: execution must be strictly after the announcement date
    la_exec = int(np.sum(mappable & (exec_d <= rdq_d)))
    la_dec = int(np.sum(mappable & (dec_d <= rdq_d)))
    # data sanity: announcement must not precede the fiscal quarter-end
    rdq_before_datadate = int((ev["rdq"] < ev["datadate"]).sum())
    # latency distribution (trading days)
    lat = ev.loc[mappable, "latency_td"]

    audit = {
        "protocol_correction": (
            "rdq sourced from w_comp_na_daily_all__fundq, NOT fundamentals_pit "
            "(v5 section 4 asserted fundamentals_pit; measured false)."
        ),
        "window_start": WINDOW_START,
        "trading_calendar": {"n_days": int(len(cal)), "start": cal_start, "end": cal_end,
                             "vendor": "polygon_ohlcv_1d"},
        "events_total": int(len(ev)),
        "gvkeys": int(ev["gvkey"].nunique()),
        "rdq_min": str(ev["rdq"].min().date()),
        "rdq_max": str(ev["rdq"].max().date()),
        "permno_link_rate": round(float(link_rate), 4),
        "permno_linked": int(ev["permno"].notna().sum()),
        "mappable_events": int(mappable.sum()),
        "unmappable_events": int((~mappable).sum()),
        "lookahead_violations_exec_le_rdq": la_exec,
        "lookahead_violations_decision_le_rdq": la_dec,
        "rdq_before_fiscal_quarter_end": rdq_before_datadate,
        "latency_td_min": int(lat.min()) if len(lat) else None,
        "latency_td_median": float(lat.median()) if len(lat) else None,
        "latency_td_max": int(lat.max()) if len(lat) else None,
        "latency_td_eq2_share": round(float((lat == 2).mean()), 4) if len(lat) else None,
        "events_by_year": {int(y): int(c) for y, c in
                           ev["rdq"].dt.year.value_counts().sort_index().items()},
        "caveat": ("rdq is date-only; conservative after-close gate applied. NOT a complete "
                   "P5a solution -- upgrade via IBES announcement times or CIQ key-developments."),
    }

    panel_cols = ["gvkey", "permno", "tic", "cusip", "cik", "conm",
                  "datadate", "fyearq", "fqtr", "rdq",
                  "decision_date", "exec_date", "latency_td", "mappable", "n_raw_rows"]
    panel = ev[panel_cols].copy()
    panel.to_parquet(OUT / "rdq_event_panel.parquet", index=False)
    (OUT / "rdq_validation.json").write_text(json.dumps(audit, indent=2))

    print("\n=== rdq validation audit ===")
    print(json.dumps({k: audit[k] for k in (
        "events_total", "gvkeys", "permno_link_rate", "mappable_events",
        "lookahead_violations_exec_le_rdq", "lookahead_violations_decision_le_rdq",
        "rdq_before_fiscal_quarter_end", "latency_td_min", "latency_td_median",
        "latency_td_max", "latency_td_eq2_share")}, indent=2))
    print(f"\n[written] {OUT/'rdq_event_panel.parquet'}")
    print(f"[written] {OUT/'rdq_validation.json'}")
    con.close()


if __name__ == "__main__":
    main()
