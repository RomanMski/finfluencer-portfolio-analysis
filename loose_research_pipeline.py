from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import math
import re

import numpy as np
import pandas as pd

from run_pipeline import classify_action_evidence


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def parse_date_series(s: pd.Series) -> pd.Series:
    # Handles yyyy-mm-dd, yyyymmdd, timestamps, etc.
    out = pd.to_datetime(s, errors="coerce")
    mask = out.isna() & s.notna()
    if mask.any():
        text = s.astype(str)
        yyyymmdd = text.str.match(r"^\d{8}$", na=False)
        if yyyymmdd.any():
            parsed = pd.to_datetime(text.where(yyyymmdd), format="%Y%m%d", errors="coerce")
            out = out.fillna(parsed)
    return out


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    mapping = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "GOOG": "GOOGL",
        "XAUUSD": "GLD",
        "XAGUSD": "SLV",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "DAX": "EWG",
        "NASDAQ": "QQQ",
        "NASDAQ100": "QQQ",
        "SP500": "SPY",
        "S&P500": "SPY",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
        "ADA": "ADA-USD",
    }
    return mapping.get(t, t)


def infer_event_type(row: pd.Series) -> str:
    action = str(row.get("action_inferred", "") or row.get("action", "") or "").lower()
    text = " ".join(str(row.get(c, "")) for c in row.index if c.lower() in {
        "quote_segment", "context_window", "text", "matched_text", "snippet", "video_title"
    }).lower()

    blob = f"{action} {text}"
    classified_action, _, _, _ = classify_action_evidence(blob)
    return {
        "buy_or_add": "buy_or_add",
        "sell_or_reduce": "sell_or_reduce",
        "hold_or_portfolio_holding": "holding_update",
        "watchlist_or_uncertain": "watchlist",
        "negative_or_not_buying": "negative_not_buying",
    }.get(classified_action, "mention_only")

def load_candidates(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "stock_pick_candidates.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_inventory(root: Path) -> pd.DataFrame:
    path = root / "data" / "video_inventory.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def attach_dates(cand: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    df = cand.copy()

    date_col = find_col(df, ["video_date", "upload_date", "date", "published_at", "timestamp"])
    if date_col:
        df["video_date"] = parse_date_series(df[date_col])
    else:
        df["video_date"] = pd.NaT

    if inv.empty:
        return df

    inv = inv.copy()
    inv_id = find_col(inv, ["video_id", "id"])
    cand_id = find_col(df, ["video_id", "id"])

    inv_date = find_col(inv, ["video_date", "upload_date", "date", "published_at", "timestamp"])
    if inv_id and cand_id and inv_date:
        inv["_video_date"] = parse_date_series(inv[inv_date])
        joined = df.merge(inv[[inv_id, "_video_date"]], left_on=cand_id, right_on=inv_id, how="left")
        joined["video_date"] = joined["video_date"].fillna(joined["_video_date"])
        return joined.drop(columns=[c for c in ["_video_date"] if c in joined.columns])

    return df


def build_events(root: Path, min_confidence: float = 0.0, include_watchlist: bool = True) -> pd.DataFrame:
    cand = load_candidates(root)
    inv = load_inventory(root)

    if cand.empty:
        return pd.DataFrame()

    df = attach_dates(cand, inv)

    ticker_col = find_col(df, ["ticker", "symbol", "asset", "asset_symbol"])
    if ticker_col is None:
        return pd.DataFrame()

    df["ticker"] = df[ticker_col].map(normalize_ticker)
    df = df[df["ticker"].notna() & df["ticker"].astype(str).str.len().gt(0)].copy()

    def classify_row(row: pd.Series) -> pd.Series:
        parts = []
        for column in ["asset_context", "context_window", "quote_segment", "video_title"]:
            value = row.get(column, "")
            if value is not None and not pd.isna(value) and str(value).strip():
                parts.append(str(value))
        text = " ".join(parts)
        action, confidence, evidence_class, strict_eligible = classify_action_evidence(text)
        return pd.Series(
            {
                "action_inferred": action,
                "event_type": {
                    "buy_or_add": "buy_or_add",
                    "sell_or_reduce": "sell_or_reduce",
                    "hold_or_portfolio_holding": "holding_update",
                    "watchlist_or_uncertain": "watchlist",
                    "negative_or_not_buying": "negative_not_buying",
                }.get(action, "mention_only"),
                "_classified_confidence": confidence,
                "evidence_class": evidence_class,
                "strict_eligible": strict_eligible,
            }
        )

    classified = df.apply(classify_row, axis=1)
    for column in classified.columns:
        df[column] = classified[column]

    confidence_col = find_col(df, ["confidence_rule_based", "confidence", "score"])
    if confidence_col:
        df["_confidence"] = pd.to_numeric(df[confidence_col], errors="coerce").fillna(0.5)
    else:
        df["_confidence"] = df["_classified_confidence"]

    # Loose mode: keep clean actions and watchlist/scenario rows, but still drop pure mentions.
    keep = ["buy_or_add", "sell_or_reduce", "holding_update"]
    if include_watchlist:
        keep.append("watchlist")

    df = df[df["event_type"].isin(keep)].copy()
    df = df[df["_confidence"] >= min_confidence].copy()

    if "video_date" not in df.columns or df["video_date"].isna().all():
        return pd.DataFrame()

    df = df.dropna(subset=["video_date"]).copy()
    df["video_date"] = pd.to_datetime(df["video_date"]).dt.normalize()
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["event_date"] = df["event_date"].fillna(df["video_date"]).dt.normalize()
    else:
        df["event_date"] = df["video_date"]

    for col in ["company", "video_title", "timestamp_url", "quote_segment", "context_window"]:
        if col not in df.columns:
            df[col] = ""

    # Prevent repeated duplicates from same video/asset/action.
    dedup_cols = ["event_date", "ticker", "event_type", "video_title"]
    df = df.drop_duplicates(subset=dedup_cols)

    return df.sort_values(["event_date", "ticker", "event_type"]).reset_index(drop=True)


def download_prices(tickers: list[str], start: str, end: str, refresh: bool, root: Path) -> pd.DataFrame:
    out = root / "data" / "market" / "adj_close.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not refresh:
        try:
            df = pd.read_csv(out, parse_dates=[0])
            df = df.set_index(df.columns[0])
        except Exception:
            df = pd.read_csv(out, parse_dates=["Date"]).set_index("Date")
        return df

    import yfinance as yf

    symbols = sorted(set([normalize_ticker(t) for t in tickers] + ["SPY", "QQQ"]))
    log(f"Downloading adjusted prices for {len(symbols)} symbols.")

    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"].copy()
        else:
            close = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        close = data.to_frame(name=symbols[0]) if isinstance(data, pd.Series) else data

    close.index = pd.to_datetime(close.index)
    close = close.dropna(axis=1, how="all").sort_index()
    close.to_csv(out, index_label="Date")
    return close


def next_trading_day(index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(pd.Timestamp(d), side="left")
    if pos >= len(index):
        return None
    return index[pos]


def build_portfolio(events: pd.DataFrame, prices: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    if events.empty or prices.empty:
        return pd.DataFrame()

    # Loose mode can display uncertain rows, but it must never turn them into trades.
    eligible = events.get("strict_eligible", pd.Series(False, index=events.index)).fillna(False)
    eligible = eligible.map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    )
    buy_events = events[events["event_type"].eq("buy_or_add") & eligible].copy()
    mode = "explicit_buy"

    if buy_events.empty:
        return pd.DataFrame()

    price_index = prices.index
    returns = prices.pct_change().fillna(0.0)

    active_by_day = {}
    entry_days = []
    for _, row in buy_events.iterrows():
        ticker = row["ticker"]
        if ticker not in prices.columns:
            continue

        entry = next_trading_day(price_index, row["event_date"] + pd.Timedelta(days=1))
        if entry is None:
            continue

        entry_pos = price_index.get_loc(entry)
        entry_days.append(entry)
        return_start_pos = entry_pos + 1
        exit_pos = min(entry_pos + int(holding_days), len(price_index) - 1)
        held_dates = price_index[return_start_pos:exit_pos + 1]

        for d in held_dates:
            active_by_day.setdefault(d, []).append(ticker)

    if not entry_days:
        return pd.DataFrame()

    rows = []
    for d in price_index:
        active = sorted(set(active_by_day.get(d, [])))
        if active:
            r = returns.loc[d, active].dropna().mean()
        else:
            r = 0.0
        rows.append({"date": d, "event_portfolio_return": r, "active_positions": len(active), "portfolio_mode": mode})

    out = pd.DataFrame(rows).set_index("date")
    if entry_days:
        out = out.loc[min(entry_days):].copy()
        out.iloc[0, out.columns.get_loc("event_portfolio_return")] = 0.0
    out["event_portfolio_growth"] = (1 + out["event_portfolio_return"]).cumprod()

    for bench in ["SPY", "QQQ"]:
        if bench in prices.columns:
            out[f"{bench}_return"] = prices[bench].pct_change().fillna(0.0)
            out.iloc[0, out.columns.get_loc(f"{bench}_return")] = 0.0
            out[f"{bench}_growth"] = (1 + out[f"{bench}_return"]).cumprod()

    return out


def perf(growth: pd.Series, label: str) -> dict:
    g = growth.dropna()
    if len(g) < 3:
        return {"series": label}

    rets = g.pct_change().dropna()
    total = g.iloc[-1] / g.iloc[0] - 1
    ann = (1 + total) ** (252 / max(len(rets), 1)) - 1
    vol = rets.std() * math.sqrt(252)
    sharpe = ann / vol if vol and not pd.isna(vol) else np.nan
    dd = g / g.cummax() - 1

    return {
        "series": label,
        "total_return": total,
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_0rf": sharpe,
        "max_drawdown": dd.min(),
    }


def build_forward_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    horizons = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}
    rows = []
    idx = prices.index

    for _, ev in events.iterrows():
        ticker = ev["ticker"]
        if ticker not in prices.columns:
            continue

        entry = next_trading_day(idx, ev["event_date"] + pd.Timedelta(days=1))
        if entry is None:
            continue
        entry_pos = idx.get_loc(entry)
        entry_px = prices.loc[entry, ticker]
        if pd.isna(entry_px) or entry_px == 0:
            continue

        row = ev.to_dict()
        row["entry_date"] = entry
        row["entry_price"] = entry_px
        for label, days in horizons.items():
            pos = entry_pos + days
            if pos < len(idx):
                px = prices.iloc[pos][ticker]
                row[f"return_{label}"] = px / entry_px - 1 if pd.notna(px) else np.nan
            else:
                row[f"return_{label}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--holding-days", type=int, default=63)
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log("Building loose event analysis from candidate rows.")

    events = build_events(root, min_confidence=args.min_confidence, include_watchlist=True)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    events_out = out_dir / "clean_candidate_events.csv"
    events.to_csv(events_out, index=False)

    if events.empty:
        log("No candidate events could be converted into a loose event dataset.")
        return

    start = (events["event_date"].min() - pd.Timedelta(days=10)).date().isoformat()
    end = (datetime.now().date() + timedelta(days=2)).isoformat()
    tickers = events["ticker"].dropna().unique().tolist()

    prices = download_prices(tickers, start, end, args.refresh_prices, root)
    if prices.empty:
        log("No price data available.")
        return

    daily = build_portfolio(events, prices, args.holding_days)
    daily_out = out_dir / "event_portfolio_daily.csv"
    daily.to_csv(daily_out)

    fwd = build_forward_returns(events, prices)
    fwd_out = out_dir / "forward_returns.csv"
    fwd.to_csv(fwd_out, index=False)

    metrics = []
    if not daily.empty and "event_portfolio_growth" in daily.columns:
        metrics.append(perf(daily["event_portfolio_growth"], "Event portfolio"))
    for bench in ["SPY", "QQQ"]:
        col = f"{bench}_growth"
        if col in daily.columns:
            metrics.append(perf(daily[col], bench))

    metrics_out = out_dir / "performance_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_out, index=False)

    summary = root / "outputs" / "presentation_summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "# Presentation summary\n\n"
        "Loose fallback analysis was used.\n\n"
        f"Events: {len(events)}\n\n"
        f"Tickers: {events['ticker'].nunique()}\n\n"
        f"Holding days: {args.holding_days}\n\n"
        "This mode is useful for general creator exploration, but the rows must be manually verified before academic use.\n",
        encoding="utf-8",
    )

    log(f"Wrote loose events: {events_out}")
    log(f"Wrote daily portfolio: {daily_out}")
    log(f"Wrote metrics: {metrics_out}")
    log("DONE loose analysis")


if __name__ == "__main__":
    main()
