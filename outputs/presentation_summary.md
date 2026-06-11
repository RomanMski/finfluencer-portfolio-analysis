# Presentation summary

## One-line project framing

We transform timestamped finance YouTube transcripts into a structured stock-recommendation dataset and test whether the extracted recommendations outperform simple market benchmarks.

## Current first-pass dataset

- Sample window: 2024-12-19 to 2026-06-08
- Candidate recommendation / portfolio-update rows after cleaning: 376
- Unique tickers: 39
- Buy/add candidates: 138
- Sell/reduce candidates: 85
- Holding-update candidates: 153

## Demo performance rule

For the first technical demo, we use a deliberately simple rule:

> Enter a stock on the next trading day after a detected buy/add event, hold it for 63 trading days, and equal-weight all active event positions. Compare the resulting event portfolio with SPY and QQQ over the same dates.

This is not the final academic portfolio yet. The final version should use manually verified recommendation rows and, if possible, actual portfolio weights shown in videos.

## Performance metrics

| series          |   total_return |   annualized_return |   annualized_volatility |   sharpe_0rf |   max_drawdown |
|:----------------|---------------:|--------------------:|------------------------:|-------------:|---------------:|
| Event portfolio |      0.0019134 |          0.00129861 |                0.207652 |      0.10899 |      -0.215966 |
| SPY             |      0.275067  |          0.179361   |                0.178659 |      1.01999 |      -0.187552 |
| QQQ             |      0.39041   |          0.250778   |                0.223473 |      1.12096 |      -0.227683 |

## Best visuals to show the group

1. `03_recommendation_timeline.png` — shows that the transcript extraction creates a real timestamped dataset.
2. `05_event_portfolio_vs_benchmarks.png` — shows the actual performance comparison against SPY and QQQ.
3. `06_drawdown_comparison.png` — shows risk, not just return.
4. `07_performance_metrics.png` — gives the clean table for the report/presentation.

## Recommended wording for the group

I got the first technical version working. It downloads/transforms the transcript data into timestamped stock-pick or portfolio-update candidates, fixes the dates, pulls real adjusted price data, and creates a first performance comparison versus SPY/QQQ. The current performance result is still a demo because the recommendation rows need manual verification, but the pipeline is useful and should save us a lot of time.
