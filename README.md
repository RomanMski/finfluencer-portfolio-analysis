# Finfluencer Portfolio Analysis

Source-backed extraction and portfolio analysis of finfluencer investment recommendations.

The app scans finance YouTube channels, downloads available transcripts, extracts stock, ETF and crypto recommendation candidates, and keeps a timestamp link and quote context for every row. When the evidence is strong enough, it builds a simple event portfolio and compares it with SPY and QQQ.

This is an academic research tool, not an investment model. Extracted events should be manually checked before they are used in the final report.

## Main features

- strict Buy/Add, Sell/Reduce and Holding extraction
- loose Watchlist and scenario fallback for less structured creators
- source fields for manual verification
- separate saved runs for every creator
- transcript, extraction and portfolio diagnostics
- event portfolio, forward returns, drawdown and benchmark analysis
- ticker price charts with recommendation markers
- optional browser-session fallback when YouTube rate-limits transcripts

Joseph Carlson is the main worked example, but the pipeline is designed for other finance channels as well.
Model-assisted extraction can be added later, but it is not configured or required by the current app.

## Install and run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Paste a channel URL or handle, choose the number of videos and start the run. Results are stored under:

```text
runs/<timestamp>_<creator>/
```

Each run keeps its own inventory, transcript status, candidate rows, market data, figures, logs and `manifest.json`. A failed creator run cannot overwrite a previous working dashboard.

## Output levels

The dashboard separates three outcomes:

1. No transcripts: shows the download failure reasons, including YouTube IP blocks.
2. Raw candidates only: shows ticker and action counts plus the timestamped source rows.
3. Event portfolio: shows performance, risk, forward returns and source-backed events.

The app does not force every creator into a portfolio. A clear diagnostic result is more useful than invented trades.

## Verification

Run the local extraction tests:

```powershell
python -m unittest discover -s tests -v
```

Run a live Joseph Carlson sanity check:

```powershell
python live_sanity_check.py --max-videos 20
```

If YouTube blocks anonymous transcript requests, the live check reports a skip with the failure count. You can also use a local browser session:

```powershell
python live_sanity_check.py --max-videos 20 --cookies-from-browser firefox
```

Close the selected browser before using its cookies if the cookie database is locked.

## Research warning

The demo portfolio enters on the next trading day after an extracted Buy/Add event, holds for a fixed number of trading days, and equal-weights active positions. The final academic analysis should use manually verified rows and clearly documented portfolio rules.

Raw full transcripts and generated run folders are intentionally excluded from Git.
