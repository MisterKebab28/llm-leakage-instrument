# Data

The pipeline reads a survivorship-free, point-in-time (PIT) equity backbone assembled from licensed
vendors. These sources are **not redistributable**; this document specifies exactly what is used so the
panels can be reconstructed from a standard WRDS subscription. The aggregate results in `results/` and
the figures in `figs/` are included so the paper's numbers can be inspected without the raw data.

## Sources and tables

| source | use | key fields |
|---|---|---|
| **CRSP** daily stock file | survivorship-free returns, delisting returns | `permno`, `date`, `ret`, `dlret` |
| **Compustat** Fundamentals Annual (`funda`) | ground-truth fundamentals (revenue) | `gvkey`, `tic`, `fyear`, `sale`, `conm` |
| **Compustat** Fundamentals Quarterly (`fundq`) | PIT earnings-event panel | `gvkey`, `rdq`, `curcdq`, `datadate` |
| **Polygon** daily OHLCV | split-adjusted prices, annual returns, the backtest | `raw_symbol`, `timestamp`, `close`, `volume` |
| CRSP/Compustat link (`ccmxpf_lnkhist`) | `gvkey` ↔ `permno` mapping | standard linktype/linkprim filters |

All Compustat pulls use the standard `datafmt='STD'`, `indfmt='INDL'`, `popsrc='D'`, `consol='C'`
filters. Annual revenue per `(tic, fyear)` is de-duplicated with `QUALIFY row_number() OVER
(PARTITION BY tic, fyear ORDER BY sale DESC) = 1`.

## Point-in-time discipline

- **Earnings events.** The event panel keys on the Compustat report date `rdq`. A decision is taken at
  the close of the first trading day strictly after `rdq` and executed at the next open; this convention
  is verified to have zero look-ahead (`rdq_01_event_panel.py`).
- **Annual returns.** Computed from the split-adjusted close, first-to-last trading day of the calendar
  year, excluding the January-1 holiday bar, in UTC (`vr_11_multicutoff_rd.py`, `vr_14_return_recall.py`).
- **Fundamentals as of a decision date.** Only fiscal years whose report date precedes the decision date
  are visible (the `seen` / `unseen` split keys on `fyear` versus the model's training cutoff).

## Models

Open-weight models are downloaded from the Hugging Face Hub (`dl_gated.py`, using `HF_TOKEN` for gated
repos) to a local cache. No-leak baselines are the ChronoGPT family (manelalab) at yearly cutoffs
2014–2024. Closed models are accessed through their public APIs. No model weights are included here.

## Reconstruction note

The scripts read a single consolidated DuckDB database (`barj_master.duckdb`) exposing the tables above
under the names referenced in `src/`. With a WRDS + Polygon subscription, materialize those tables (or
adjust the table names in `src/`) and the pipeline runs end-to-end.
