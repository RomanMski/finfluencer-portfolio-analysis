from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


BENCHMARKS = ["SPY", "QQQ"]
HORIZONS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{proc.stderr[:4000]}")
    return proc.stdout


def normalize_video_id_from_url(url: str) -> str:
    url = str(url)
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{8,})", url)
    return m.group(1) if m else ""


def normalize_date(upload_date: Any, timestamp: Any = None) -> str:
    if upload_date is not None and str(upload_date).strip() and str(upload_date) != "nan":
        s = str(upload_date).strip()
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s

    if timestamp not in [None, "", np.nan]:
        try:
            return datetime.utcfromtimestamp(float(timestamp)).strftime("%Y-%m-%d")
        except Exception:
            return ""

    return ""


def collect_or_update_inventory(channel: str, max_videos: int, data_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    Collects video inventory with full metadata. This is slower than flat playlist mode but usually gives upload_date.
    """
    inventory_path = data_dir / "video_inventory.csv"

    if inventory_path.exists() and not force:
        inv = pd.read_csv(inventory_path)
        if "video_id" not in inv.columns:
            inv["video_id"] = inv.get("url", "").map(normalize_video_id_from_url)
        missing_dates = inv.get("date", pd.Series([""] * len(inv))).isna() | (inv.get("date", "") == "")
        if len(inv) > 0 and missing_dates.mean() < 0.10:
            log(f"Using existing inventory: {inventory_path}")
            return inv

    log("Collecting full YouTube metadata with yt-dlp. This can take a few minutes.")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--dump-json",
        "--skip-download",
        "--playlist-end",
        str(max_videos),
        "--ignore-errors",
        channel,
    ]

    raw = run_cmd(cmd)
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue

        video_id = item.get("id", "")
        if not video_id:
            continue

        date = normalize_date(item.get("upload_date"), item.get("timestamp"))
        rows.append(
            {
                "video_id": video_id,
                "date": date,
                "title": item.get("title", ""),
                "url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                "duration": item.get("duration", ""),
                "view_count": item.get("view_count", ""),
                "channel": item.get("channel", ""),
                "upload_date_raw": item.get("upload_date", ""),
                "timestamp_raw": item.get("timestamp", ""),
            }
        )

    inv = pd.DataFrame(rows).drop_duplicates(subset=["video_id"])
    data_dir.mkdir(parents=True, exist_ok=True)
    inv.to_csv(inventory_path, index=False)
    log(f"Saved inventory with {len(inv)} videos to {inventory_path}")
    return inv


def fix_candidate_dates(candidates: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    if "video_id" not in df.columns:
        df["video_id"] = df["video_url"].map(normalize_video_id_from_url)

    inv = inventory.copy()
    if "video_id" not in inv.columns:
        inv["video_id"] = inv["url"].map(normalize_video_id_from_url)

    inv_small = inv[["video_id", "date", "title", "url"]].drop_duplicates("video_id").rename(
        columns={"date": "inventory_date", "title": "inventory_title", "url": "inventory_url"}
    )

    df = df.merge(inv_small, on="video_id", how="left")

    if "video_date" not in df.columns:
        df["video_date"] = ""

    df["video_date"] = df["video_date"].fillna("")
    df["video_date"] = np.where(df["video_date"].astype(str).str.len() >= 8, df["video_date"], df["inventory_date"].fillna(""))
    df["video_title"] = df.get("video_title", "").fillna("")
    df["video_title"] = np.where(df["video_title"].astype(str).str.len() > 0, df["video_title"], df["inventory_title"].fillna(""))
    df["video_url"] = df.get("video_url", "").fillna("")
    df["video_url"] = np.where(df["video_url"].astype(str).str.len() > 0, df["video_url"], df["inventory_url"].fillna(""))

    df = df.drop(columns=[c for c in ["inventory_date", "inventory_title", "inventory_url"] if c in df.columns])
    return df


def normalize_ticker_for_yfinance(ticker: str) -> str:
    t = str(ticker).strip().upper()
    mapping = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "GOOG": "GOOGL",   # collapse Google class shares for cleaner charts
    }
    return mapping.get(t, t)


def clean_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()

    df["video_date"] = pd.to_datetime(df["video_date"], errors="coerce")
    df = df.dropna(subset=["video_date"])
    df["ticker_raw"] = df["ticker"].astype(str).str.upper().str.strip()
    df["ticker"] = df["ticker_raw"].map(normalize_ticker_for_yfinance)

    keep_actions = ["buy_or_add", "sell_or_reduce", "hold_or_portfolio_holding"]
    df = df[df["action_inferred"].isin(keep_actions)].copy()

    if "confidence_rule_based" in df.columns:
        df["confidence_rule_based"] = pd.to_numeric(df["confidence_rule_based"], errors="coerce").fillna(0)
        df = df[df["confidence_rule_based"] >= 0.55].copy()

    # Remove duplicate same ticker/action from same video. Keep the earliest timestamp.
    df = df.sort_values(["video_date", "video_url", "ticker", "timestamp_start_seconds"])
    df = df.drop_duplicates(subset=["video_url", "ticker", "action_inferred"], keep="first")

    df["event_type"] = np.where(
        df["action_inferred"].eq("sell_or_reduce"),
        "sell_or_reduce",
        np.where(df["action_inferred"].eq("buy_or_add"), "buy_or_add", "holding_update"),
    )

    # Portfolio demo uses only clear buy/add rows by default.
    df["use_for_demo_portfolio"] = df["event_type"].eq("buy_or_add")
    df["manual_verification_needed"] = True

    cols = [
        "video_date",
        "video_title",
        "video_url",
        "timestamp_start",
        "timestamp_end",
        "timestamp_url",
        "ticker",
        "ticker_raw",
        "company",
        "event_type",
        "action_inferred",
        "confidence_rule_based",
        "quote_segment",
        "context_window",
        "use_for_demo_portfolio",
        "manual_verification_needed",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].reset_index(drop=True)


def download_prices(tickers: list[str], start: str, end: str, out_csv: Path, refresh: bool = False) -> pd.DataFrame:
    if out_csv.exists() and not refresh:
        log(f"Using existing price file: {out_csv}")
        px = pd.read_csv(out_csv, index_col=0, parse_dates=True)
        return px

    import yfinance as yf

    all_tickers = sorted(set(tickers + BENCHMARKS))
    log(f"Downloading adjusted prices for {len(all_tickers)} tickers from yfinance.")
    raw = yf.download(
        tickers=all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No price data downloaded. Check internet connection/yfinance access.")

    if isinstance(raw.columns, pd.MultiIndex):
        close_cols = []
        for t in all_tickers:
            if (t, "Close") in raw.columns:
                close_cols.append((t, "Close"))
        px = raw[close_cols].copy()
        px.columns = [c[0] for c in px.columns]
    else:
        # Single ticker fallback
        px = raw[["Close"]].copy()
        px.columns = [all_tickers[0]]

    px = px.dropna(axis=1, how="all")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(out_csv)
    return px


def nearest_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    if pd.isna(date):
        return None
    pos = index.searchsorted(pd.Timestamp(date).normalize(), side="left")
    if pos >= len(index):
        return None
    return index[pos]


def compute_forward_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = prices.index

    for _, ev in events.iterrows():
        ticker = ev["ticker"]
        if ticker not in prices.columns:
            continue
        start_day = nearest_trading_day(idx, ev["video_date"] + pd.Timedelta(days=1))
        if start_day is None:
            continue
        p0 = prices.loc[start_day, ticker]
        if pd.isna(p0) or p0 <= 0:
            continue

        row = ev.to_dict()
        row["entry_date"] = start_day.date().isoformat()
        row["entry_price"] = float(p0)

        for label, days in HORIZONS.items():
            start_pos = idx.get_loc(start_day)
            end_pos = min(start_pos + days, len(idx) - 1)
            end_day = idx[end_pos]
            p1 = prices.iloc[end_pos][ticker]
            row[f"return_{label}"] = np.nan if pd.isna(p1) else float(p1 / p0 - 1.0)
            row[f"end_date_{label}"] = end_day.date().isoformat()

        rows.append(row)

    return pd.DataFrame(rows)


def build_event_portfolio(events: pd.DataFrame, prices: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    """
    Demonstration portfolio:
    - uses only buy/add candidate events
    - enters next trading day after video upload
    - holds each signal for `holding_days` trading days
    - equal-weights unique active tickers each day
    """
    idx = prices.index
    returns = prices.pct_change().fillna(0)

    long_events = events[events["use_for_demo_portfolio"]].copy()
    if long_events.empty:
        raise RuntimeError("No buy/add events available for portfolio demo.")

    active_by_day = {d: set() for d in idx}

    for _, ev in long_events.iterrows():
        ticker = ev["ticker"]
        if ticker not in prices.columns:
            continue

        entry_day = nearest_trading_day(idx, ev["video_date"] + pd.Timedelta(days=1))
        if entry_day is None:
            continue
        start_pos = idx.get_loc(entry_day)
        end_pos = min(start_pos + holding_days, len(idx) - 1)

        for d in idx[start_pos:end_pos + 1]:
            if not pd.isna(prices.loc[d, ticker]):
                active_by_day[d].add(ticker)

    rows = []
    for d in idx:
        active = sorted(active_by_day[d])
        if active:
            r = returns.loc[d, active].mean()
        else:
            r = 0.0

        rows.append(
            {
                "date": d,
                "event_portfolio_return": float(r),
                "active_tickers": ",".join(active),
                "n_active": len(active),
                "SPY_return": float(returns.loc[d, "SPY"]) if "SPY" in returns.columns else np.nan,
                "QQQ_return": float(returns.loc[d, "QQQ"]) if "QQQ" in returns.columns else np.nan,
            }
        )

    daily = pd.DataFrame(rows).set_index("date")
    # Trim to period with at least one active position
    if daily["n_active"].gt(0).any():
        first = daily.index[daily["n_active"].gt(0)][0]
        daily = daily.loc[first:].copy()

    for c in ["event_portfolio_return", "SPY_return", "QQQ_return"]:
        if c in daily.columns:
            daily[c.replace("_return", "_growth")] = (1 + daily[c].fillna(0)).cumprod()

    return daily


def max_drawdown(growth: pd.Series) -> float:
    growth = growth.dropna()
    if growth.empty:
        return np.nan
    peak = growth.cummax()
    dd = growth / peak - 1
    return float(dd.min())


def annualized_return(growth: pd.Series) -> float:
    growth = growth.dropna()
    if len(growth) < 2:
        return np.nan
    years = (growth.index[-1] - growth.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float(growth.iloc[-1] ** (1 / years) - 1)


def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std() * np.sqrt(252))


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Event portfolio", "event_portfolio_return", "event_portfolio_growth"),
        ("SPY", "SPY_return", "SPY_growth"),
        ("QQQ", "QQQ_return", "QQQ_growth"),
    ]

    rows = []
    for name, rcol, gcol in specs:
        if rcol not in daily.columns or gcol not in daily.columns:
            continue
        rows.append(
            {
                "series": name,
                "total_return": float(daily[gcol].iloc[-1] - 1),
                "annualized_return": annualized_return(daily[gcol]),
                "annualized_volatility": annualized_vol(daily[rcol]),
                "sharpe_0rf": sharpe(daily[rcol]),
                "max_drawdown": max_drawdown(daily[gcol]),
            }
        )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.1%}"


def make_figures(events: pd.DataFrame, fwd: pd.DataFrame, daily: pd.DataFrame, metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 01 action distribution
    action_counts = events["event_type"].value_counts().sort_values()
    plt.figure(figsize=(9, 5))
    action_counts.plot(kind="barh")
    plt.title("Extracted recommendation candidates by type")
    plt.xlabel("Verified-needed candidate rows")
    plt.tight_layout()
    plt.savefig(fig_dir / "01_action_distribution.png", dpi=180)
    plt.close()

    # 02 top tickers
    top = events["ticker"].value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    top.plot(kind="barh")
    plt.title("Most frequent tickers in extracted candidates")
    plt.xlabel("Candidate rows")
    plt.tight_layout()
    plt.savefig(fig_dir / "02_top_tickers.png", dpi=180)
    plt.close()

    # 03 timeline
    top_tickers = events["ticker"].value_counts().head(12).index.tolist()
    tl = events[events["ticker"].isin(top_tickers)].copy()
    ymap = {t: i for i, t in enumerate(reversed(top_tickers))}
    tl["y"] = tl["ticker"].map(ymap)
    plt.figure(figsize=(11, 6))
    for etype, g in tl.groupby("event_type"):
        plt.scatter(g["video_date"], g["y"], label=etype, alpha=0.8, s=35)
    plt.yticks(list(ymap.values()), list(ymap.keys()))
    plt.title("Timestamped recommendation timeline")
    plt.xlabel("Video upload date")
    plt.ylabel("Ticker")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "03_recommendation_timeline.png", dpi=180)
    plt.close()

    # 04 forward returns
    if not fwd.empty:
        long = []
        for h in HORIZONS.keys():
            col = f"return_{h}"
            if col in fwd.columns:
                temp = fwd[["ticker", "event_type", col]].dropna().copy()
                temp["horizon"] = h
                temp["return"] = temp[col]
                long.append(temp[["ticker", "event_type", "horizon", "return"]])
        if long:
            fr = pd.concat(long, ignore_index=True)
            avg = fr.groupby(["horizon", "event_type"])["return"].mean().reset_index()
            pivot = avg.pivot(index="horizon", columns="event_type", values="return").reindex(list(HORIZONS.keys()))
            plt.figure(figsize=(9, 5))
            pivot.plot(kind="bar", ax=plt.gca())
            plt.axhline(0, linewidth=1)
            plt.title("Average forward returns after extracted events")
            plt.xlabel("Forward horizon")
            plt.ylabel("Average return")
            plt.tight_layout()
            plt.savefig(fig_dir / "04_forward_returns_by_horizon.png", dpi=180)
            plt.close()

    # 05 cumulative performance
    plt.figure(figsize=(11, 6))
    for col, label in [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]:
        if col in daily.columns:
            plt.plot(daily.index, daily[col], label=label)
    plt.title("Demo event portfolio vs benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "05_event_portfolio_vs_benchmarks.png", dpi=180)
    plt.close()

    # 06 drawdown comparison
    plt.figure(figsize=(11, 6))
    for col, label in [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]:
        if col in daily.columns:
            growth = daily[col].dropna()
            dd = growth / growth.cummax() - 1
            plt.plot(dd.index, dd, label=label)
    plt.title("Drawdown comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "06_drawdown_comparison.png", dpi=180)
    plt.close()

    # 07 metrics table as visual
    if not metrics.empty:
        m = metrics.copy()
        display_cols = ["series", "total_return", "annualized_return", "annualized_volatility", "sharpe_0rf", "max_drawdown"]
        m = m[display_cols]
        for c in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
            m[c] = m[c].map(pct)
        m["sharpe_0rf"] = m["sharpe_0rf"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.axis("off")
        table = ax.table(
            cellText=m.values,
            colLabels=["Series", "Total return", "Ann. return", "Ann. vol", "Sharpe", "Max drawdown"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title("Performance summary")
        plt.tight_layout()
        plt.savefig(fig_dir / "07_performance_metrics.png", dpi=180)
        plt.close()


def write_presentation_summary(events: pd.DataFrame, fwd: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, holding_days: int) -> None:
    start = events["video_date"].min().date().isoformat() if not events.empty else ""
    end = events["video_date"].max().date().isoformat() if not events.empty else ""

    n_events = len(events)
    n_buy = int((events["event_type"] == "buy_or_add").sum()) if not events.empty else 0
    n_sell = int((events["event_type"] == "sell_or_reduce").sum()) if not events.empty else 0
    n_hold = int((events["event_type"] == "holding_update").sum()) if not events.empty else 0
    n_tickers = events["ticker"].nunique() if not events.empty else 0

    metric_text = metrics.to_markdown(index=False) if not metrics.empty else "No metrics available."

    text = f"""# Presentation summary

## One-line project framing

We transform timestamped finance YouTube transcripts into a structured stock-recommendation dataset and test whether the extracted recommendations outperform simple market benchmarks.

## Current first-pass dataset

- Sample window: {start} to {end}
- Candidate recommendation / portfolio-update rows after cleaning: {n_events}
- Unique tickers: {n_tickers}
- Buy/add candidates: {n_buy}
- Sell/reduce candidates: {n_sell}
- Holding-update candidates: {n_hold}

## Demo performance rule

For the first technical demo, we use a deliberately simple rule:

> Enter a stock on the next trading day after a detected buy/add event, hold it for {holding_days} trading days, and equal-weight all active event positions. Compare the resulting event portfolio with SPY and QQQ over the same dates.

This is not the final academic portfolio yet. The final version should use manually verified recommendation rows and, if possible, actual portfolio weights shown in videos.

## Performance metrics

{metric_text}

## Best visuals to show the group

1. `03_recommendation_timeline.png` — shows that the transcript extraction creates a real timestamped dataset.
2. `05_event_portfolio_vs_benchmarks.png` — shows the actual performance comparison against SPY and QQQ.
3. `06_drawdown_comparison.png` — shows risk, not just return.
4. `07_performance_metrics.png` — gives the clean table for the report/presentation.

## Recommended wording for the group

I got the first technical version working. It downloads/transforms the transcript data into timestamped stock-pick or portfolio-update candidates, fixes the dates, pulls real adjusted price data, and creates a first performance comparison versus SPY/QQQ. The current performance result is still a demo because the recommendation rows need manual verification, but the pipeline is useful and should save us a lot of time.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="https://www.youtube.com/@JosephCarlsonShow/videos")
    ap.add_argument("--max-videos", type=int, default=80)
    ap.add_argument("--holding-days", type=int, default=63)
    ap.add_argument("--refresh-inventory", action="store_true")
    ap.add_argument("--refresh-prices", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    market_dir = data_dir / "market"
    fig_dir = root / "outputs" / "figures"

    candidates_path = processed_dir / "stock_pick_candidates.csv"
    if not candidates_path.exists():
        # fallback: maybe user is in first generated project structure
        alt = root / "stock_pick_candidates.csv"
        if alt.exists():
            candidates_path = alt
        else:
            raise FileNotFoundError(
                f"Could not find {processed_dir / 'stock_pick_candidates.csv'}. "
                "Run the first transcript extraction pipeline first."
            )

    log("Loading candidate rows.")
    candidates = pd.read_csv(candidates_path)

    inventory = collect_or_update_inventory(args.channel, args.max_videos, data_dir, force=args.refresh_inventory)

    log("Fixing missing candidate dates.")
    dated = fix_candidate_dates(candidates, inventory)
    dated_out = processed_dir / "stock_pick_candidates_with_dates.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dated.to_csv(dated_out, index=False)

    log("Cleaning candidate rows.")
    events = clean_candidates(dated)
    events_out = processed_dir / "clean_candidate_events.csv"
    events.to_csv(events_out, index=False)

    if events.empty:
        raise RuntimeError(
            "No clean candidate events after filtering. Check whether video dates were fixed and candidates exist."
        )

    start_date = (events["video_date"].min() - pd.Timedelta(days=5)).date().isoformat()
    end_date = (datetime.today() + timedelta(days=7)).date().isoformat()
    tickers = sorted(events["ticker"].dropna().unique().tolist())

    prices = download_prices(tickers, start_date, end_date, market_dir / "adj_close.csv", refresh=args.refresh_prices)

    log("Computing forward returns.")
    fwd = compute_forward_returns(events, prices)
    fwd_out = processed_dir / "forward_returns.csv"
    fwd.to_csv(fwd_out, index=False)

    log("Building demo event portfolio.")
    daily = build_event_portfolio(events, prices, holding_days=args.holding_days)
    daily_out = processed_dir / "event_portfolio_daily.csv"
    daily.to_csv(daily_out)

    log("Computing performance metrics.")
    metrics = compute_metrics(daily)
    metrics_out = processed_dir / "performance_metrics.csv"
    metrics.to_csv(metrics_out, index=False)

    log("Creating presentation figures.")
    make_figures(events, fwd, daily, metrics, fig_dir)

    log("Writing presentation summary.")
    write_presentation_summary(events, fwd, metrics, root / "outputs" / "presentation_summary.md", args.holding_days)

    print("\nDONE V2")
    print(f"Clean events:      {events_out}")
    print(f"Forward returns:   {fwd_out}")
    print(f"Daily portfolio:   {daily_out}")
    print(f"Metrics:           {metrics_out}")
    print(f"Figures:           {fig_dir}")
    print(f"Summary:           {root / 'outputs' / 'presentation_summary.md'}")


if __name__ == "__main__":
    main()
