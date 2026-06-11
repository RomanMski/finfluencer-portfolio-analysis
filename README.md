# Finfluencer Portfolio Extraction and Performance Analysis

A reproducible data-science pipeline for a finance seminar project.

The pipeline turns timestamped YouTube transcript data into a structured stock-recommendation dataset and then runs a first empirical performance analysis.

## What it does

1. Downloads or updates YouTube video metadata.
2. Fixes missing upload dates in extracted recommendation rows.
3. Cleans candidate stock-pick / portfolio-update rows.
4. Downloads real adjusted stock prices with `yfinance`.
5. Computes forward returns after recommendations.
6. Builds a simple equal-weighted event portfolio.
7. Compares the result with SPY and QQQ over the same timeframe.
8. Saves CSV files, charts, and a short presentation summary.

## Important academic note

The generated dataset is a first-pass candidate extraction. For the final report, manually verify all rows used in the portfolio construction by opening the timestamp URL and checking whether the influencer actually recommends, buys, sells, or holds the stock.

Do not publicly commit raw full transcripts. Keep them private or regenerate them locally.

## Setup

Windows PowerShell:

```powershell
cd $env:USERPROFILE\Downloads\joseph_carlson_finfluencer_pipeline

# If the venv already exists from the first run:
.\.venv\Scripts\Activate.ps1

# Install the new dependencies:
pip install -r requirements_v2.txt

# Run full analysis:
python .\run_full_research_pipeline.py --channel "https://www.youtube.com/@JosephCarlsonShow/videos" --max-videos 80 --holding-days 63
```

## Main outputs

```text
data\processed\clean_candidate_events.csv
data\processed\forward_returns.csv
data\processed\event_portfolio_daily.csv
data\market\adj_close.csv
outputs\figures\01_action_distribution.png
outputs\figures\02_top_tickers.png
outputs\figures\03_recommendation_timeline.png
outputs\figures\04_forward_returns_by_horizon.png
outputs\figures\05_event_portfolio_vs_benchmarks.png
outputs\figures\06_drawdown_comparison.png
outputs\figures\07_performance_metrics.png
outputs\presentation_summary.md
```

## Suggested GitHub repo framing

Repository name:

```text
finfluencer-portfolio-analysis
```

Description:

```text
Transcript-to-portfolio pipeline for analysing stock recommendations by finance YouTube channels.
```

Keep the repo private while working with the group. After grading, publish a cleaned version without raw transcripts.
