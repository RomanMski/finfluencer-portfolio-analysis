from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"

st.set_page_config(page_title="Finfluencer Portfolio Analysis", layout="wide")


def normalize_channel_input(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        value = value.rstrip("/")
        if "youtube.com/@" in value and not value.endswith("/videos"):
            value += "/videos"
        return value
    handle = value if value.startswith("@") else f"@{value}"
    return f"https://www.youtube.com/{handle}/videos"


def creator_slug(channel: str) -> str:
    match = re.search(r"youtube\.com/@([^/]+)", channel, flags=re.IGNORECASE)
    base = match.group(1) if match else channel
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-").lower()[:50]
    digest = hashlib.md5(channel.encode("utf-8")).hexdigest()[:6]
    return f"{base or 'creator'}-{digest}"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def read_manifest(run_dir: Path) -> dict:
    try:
        return json.loads(manifest_path(run_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_manifest(run_dir: Path, **updates) -> None:
    manifest = read_manifest(run_dir)
    manifest.update(updates)
    manifest_path(run_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_run(channel: str, max_videos: int, holding_days: int, requested_mode: str) -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_ROOT / f"{stamp}_{creator_slug(channel)}"
    for relative in ["data/transcripts", "data/processed", "data/market", "outputs", "logs"]:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    update_manifest(
        run_dir,
        channel=channel,
        max_videos=max_videos,
        holding_days=holding_days,
        requested_mode=requested_mode,
        mode="running",
        ok=False,
        message="Run started",
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    return run_dir


def discover_runs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    runs = [path for path in RUNS_ROOT.iterdir() if path.is_dir() and manifest_path(path).exists()]
    return sorted(runs, reverse=True)


def run_label(run_dir: Path) -> str:
    manifest = read_manifest(run_dir)
    channel = manifest.get("channel", run_dir.name)
    handle = channel.replace("https://www.youtube.com/", "").replace("/videos", "")
    created = manifest.get("created_at", "")
    return f"{created} | {handle} | {manifest.get('mode', 'run')}"


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return 0


def run_stats(run_dir: Path) -> dict:
    status = load_csv(run_dir / "data/transcript_status.csv")
    inventory = load_csv(run_dir / "data/video_inventory.csv")
    success = 0
    failures = 0
    if not status.empty and "transcript_status" in status.columns:
        success = int(status["transcript_status"].isin(["ok", "already_exists"]).sum())
        failures = int(status["transcript_status"].eq("failed").sum())
    return {
        "videos": len(inventory),
        "transcripts": success,
        "transcript_failures": failures,
        "candidates": csv_row_count(run_dir / "data/processed/stock_pick_candidates.csv"),
        "events": csv_row_count(run_dir / "data/processed/clean_candidate_events.csv"),
        "portfolio_days": csv_row_count(run_dir / "data/processed/event_portfolio_daily.csv"),
    }


def parse_progress(line: str) -> dict | None:
    match = re.search(
        r"PROGRESS transcripts (\d+) (\d+) elapsed=(\d+) eta=(\d+)",
        line,
    )
    if not match:
        return None
    return {
        "done": int(match.group(1)),
        "total": int(match.group(2)),
        "elapsed": int(match.group(3)),
        "eta": int(match.group(4)),
    }


def format_seconds(seconds: int) -> str:
    minutes, remaining = divmod(max(seconds, 0), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining}s"
    return f"{remaining}s"


def run_step(
    command: list[str],
    label: str,
    start_fraction: float,
    end_fraction: float,
    log_path: Path,
) -> tuple[int, str]:
    progress = st.progress(start_fraction, text=label)
    status_box = st.empty()
    metrics = st.columns(4)
    metric_boxes = [column.empty() for column in metrics]
    with st.expander(f"{label} technical log", expanded=False):
        log_box = st.empty()

    lines = []
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if not line:
            continue
        lines.append(line)
        log_box.code("\n".join(lines[-80:]))
        parsed = parse_progress(line)
        if parsed:
            ratio = parsed["done"] / max(parsed["total"], 1)
            fraction = start_fraction + (end_fraction - start_fraction) * ratio
            progress.progress(min(fraction, end_fraction), text=f"{label}: {parsed['done']} / {parsed['total']} videos")
            status_box.info(f"Processing video {parsed['done']} of {parsed['total']}")
            metric_boxes[0].metric("Step", label)
            metric_boxes[1].metric("Videos", f"{parsed['done']} / {parsed['total']}")
            metric_boxes[2].metric("Elapsed", format_seconds(parsed["elapsed"]))
            metric_boxes[3].metric("ETA", format_seconds(parsed["eta"]))

    code = process.wait()
    elapsed = int(time.monotonic() - started)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    if code == 0:
        progress.progress(end_fraction, text=f"{label} finished")
        status_box.success(f"{label} finished in {format_seconds(elapsed)}")
    else:
        status_box.error(f"{label} failed after {format_seconds(elapsed)}")
    return code, "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=list(date_columns))
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def load_run_data(run_dir: Path) -> dict[str, pd.DataFrame]:
    data = {
        "inventory": load_csv(run_dir / "data/video_inventory.csv"),
        "status": load_csv(run_dir / "data/transcript_status.csv"),
        "candidates": load_csv(run_dir / "data/processed/stock_pick_candidates.csv"),
        "events": load_csv(run_dir / "data/processed/clean_candidate_events.csv", ("video_date",)),
        "forward": load_csv(run_dir / "data/processed/forward_returns.csv", ("video_date",)),
        "daily": load_csv(run_dir / "data/processed/event_portfolio_daily.csv", ("date",)),
        "metrics": load_csv(run_dir / "data/processed/performance_metrics.csv"),
        "prices": load_csv(run_dir / "data/market/adj_close.csv", ("Date",)),
    }
    if not data["daily"].empty and "date" in data["daily"].columns:
        data["daily"] = data["daily"].set_index("date").sort_index()
    if not data["prices"].empty:
        date_col = data["prices"].columns[0]
        data["prices"] = data["prices"].set_index(date_col)
        data["prices"].index = pd.to_datetime(data["prices"].index, errors="coerce")
        data["prices"] = data["prices"][~data["prices"].index.isna()].sort_index()
    return data


def percent(value, decimals: int = 1) -> str:
    try:
        return "" if pd.isna(value) else f"{float(value):.{decimals}%}"
    except Exception:
        return ""


def number(value, decimals: int = 2) -> str:
    try:
        return "" if pd.isna(value) else f"{float(value):.{decimals}f}"
    except Exception:
        return ""


def metric_value(metrics: pd.DataFrame, series: str, column: str):
    if metrics.empty or "series" not in metrics.columns or column not in metrics.columns:
        return np.nan
    row = metrics[metrics["series"].eq(series)]
    return np.nan if row.empty else row.iloc[0][column]


def mentioned_buy_hold_growth(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.Series:
    if prices.empty or events.empty:
        return pd.Series(dtype=float)
    available = sorted(set(events["ticker"].dropna()).intersection(prices.columns))
    if not available:
        return pd.Series(dtype=float)
    start_date = pd.Timestamp(events["video_date"].min())
    frame = prices[available].loc[
        (prices.index >= start_date) & (prices.index <= pd.Timestamp(end_date))
    ]
    frame = frame.dropna(axis=1, how="all")
    if frame.empty:
        return pd.Series(dtype=float)
    normalized = frame.apply(
        lambda series: series / series.dropna().iloc[0] if not series.dropna().empty else series,
        axis=0,
    )
    return normalized.mean(axis=1).dropna()


def growth_chart(
    daily: pd.DataFrame,
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    for column, label in [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]:
        if column in daily.columns:
            figure.add_trace(go.Scatter(x=daily.index, y=daily[column], mode="lines", name=label))
    if not daily.empty:
        mentioned_growth = mentioned_buy_hold_growth(prices, events, daily.index.max())
        if not mentioned_growth.empty:
            figure.add_trace(
                go.Scatter(
                    x=mentioned_growth.index,
                    y=mentioned_growth,
                    mode="lines",
                    name="Mentioned tickers buy and hold",
                )
            )
    figure.update_layout(
        title="Portfolio performance",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        height=520,
    )
    return figure


def drawdown_chart(daily: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, label in [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]:
        if column in daily.columns:
            growth = daily[column].dropna()
            drawdown = growth / growth.cummax() - 1
            figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name=label))
    figure.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return figure


def event_timeline(events: pd.DataFrame) -> go.Figure:
    top = events["ticker"].value_counts().head(20).index
    frame = events[events["ticker"].isin(top)].copy()
    return px.scatter(
        frame,
        x="video_date",
        y="ticker",
        color="event_type",
        hover_data=["video_title", "timestamp_url", "quote_segment"],
        title="Source-backed event timeline",
        template="plotly_white",
        height=540,
    )


def forward_return_chart(forward: pd.DataFrame) -> go.Figure:
    return_columns = [column for column in forward.columns if column.startswith("return_")]
    if forward.empty or not return_columns:
        return go.Figure()
    parts = []
    for column in return_columns:
        frame = forward[["event_type", column]].copy()
        frame["horizon"] = column.replace("return_", "")
        frame["return"] = pd.to_numeric(frame[column], errors="coerce")
        parts.append(frame[["event_type", "horizon", "return"]])
    averages = (
        pd.concat(parts, ignore_index=True)
        .dropna()
        .groupby(["horizon", "event_type"], as_index=False)["return"]
        .mean()
    )
    figure = px.bar(
        averages,
        x="horizon",
        y="return",
        color="event_type",
        barmode="group",
        title="Average forward returns after extracted events",
        template="plotly_white",
        height=430,
    )
    figure.update_layout(yaxis_tickformat=".0%")
    return figure


def ticker_chart(events: pd.DataFrame, prices: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    if ticker in prices.columns:
        series = prices[ticker].dropna()
        figure.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=ticker))
        ticker_events = events[events["ticker"].eq(ticker)]
        marker_map = {
            "buy_or_add": ("triangle-up", "Buy/add"),
            "sell_or_reduce": ("triangle-down", "Sell/reduce"),
            "holding_update": ("circle", "Holding update"),
            "watchlist": ("diamond", "Watchlist"),
        }
        for event_type, (symbol, label) in marker_map.items():
            rows = ticker_events[ticker_events["event_type"].eq(event_type)]
            xs, ys, texts = [], [], []
            for _, row in rows.iterrows():
                position = series.index.searchsorted(pd.Timestamp(row["video_date"]), side="left")
                if position >= len(series):
                    continue
                date = series.index[position]
                xs.append(date)
                ys.append(series.iloc[position])
                texts.append(f"{row.get('video_title', '')}<br>{row.get('quote_segment', '')}")
            if xs:
                figure.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=label,
                        marker={"size": 12, "symbol": symbol},
                        text=texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
    figure.update_layout(
        title=f"{ticker}: price with extracted events",
        xaxis_title="Date",
        yaxis_title="Adjusted close",
        template="plotly_white",
        height=520,
    )
    return figure


def show_diagnostics(run_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    stats = run_stats(run_dir)
    columns = st.columns(5)
    columns[0].metric("Videos scanned", stats["videos"])
    columns[1].metric("Transcripts", stats["transcripts"])
    columns[2].metric("Raw candidates", stats["candidates"])
    columns[3].metric("Clean events", stats["events"])
    columns[4].metric("Portfolio days", stats["portfolio_days"])

    status = data["status"]
    if not status.empty and "error_type" in status.columns:
        failures = status[status["transcript_status"].eq("failed")]
        if not failures.empty:
            counts = failures["error_type"].fillna("download_failed").replace("", "download_failed").value_counts()
            if "ip_blocked" in counts.index:
                st.warning(
                    "YouTube blocked transcript requests from this IP. Try a browser session in the run form, "
                    "close that browser first if its cookie database is locked, or retry later."
                )
            st.dataframe(
                counts.rename_axis("failure_reason").reset_index(name="videos"),
                hide_index=True,
                width="stretch",
            )

    candidates = data["candidates"]
    if not candidates.empty:
        left, right = st.columns(2)
        if "action_inferred" in candidates.columns:
            action_counts = candidates["action_inferred"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            left.plotly_chart(
                px.bar(
                    action_counts,
                    x="count",
                    y="action",
                    orientation="h",
                    title="Raw candidate actions",
                    template="plotly_white",
                ),
                width="stretch",
            )
        if "ticker" in candidates.columns:
            ticker_counts = candidates["ticker"].value_counts().head(20).reset_index()
            ticker_counts.columns = ["ticker", "count"]
            right.plotly_chart(
                px.bar(
                    ticker_counts,
                    x="count",
                    y="ticker",
                    orientation="h",
                    title="Top candidate tickers",
                    template="plotly_white",
                ),
                width="stretch",
            )


def source_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "video_id",
        "video_date",
        "video_title",
        "ticker",
        "company",
        "event_type",
        "action_inferred",
        "timestamp_seconds",
        "timestamp_url",
        "quote_segment",
        "context_window",
        "confidence_rule_based",
        "source_method",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


st.title("Finfluencer Portfolio Analysis")
st.caption("Source-backed extraction and portfolio analysis of finfluencer investment recommendations.")

with st.container(border=True):
    st.subheader("Run channel analysis")
    first, second, third = st.columns([2.4, 0.7, 0.7])
    channel_raw = first.text_input(
        "YouTube channel URL or handle",
        value="https://www.youtube.com/@JosephCarlsonShow/videos",
    )
    max_videos = second.number_input("Videos", min_value=10, max_value=300, value=30, step=10)
    holding_days = third.number_input("Hold days", min_value=5, max_value=252, value=63, step=5)
    requested_mode = st.selectbox(
        "Analysis mode",
        ["Strict portfolio mode", "Loose exploration fallback"],
        index=1,
    )
    st.caption("Optional model-assisted extraction is not configured. The default extractor uses deterministic rules and needs no API key.")
    cookie_choice = st.selectbox(
        "YouTube browser session (optional)",
        ["None", "chrome", "edge", "firefox", "brave"],
        help="Useful when YouTube rate-limits anonymous transcript requests. Close the selected browser before running if its cookie database is locked.",
    )
    refresh_prices = st.checkbox("Refresh market prices", value=False)

    if st.button("Run analysis", type="primary", width="stretch"):
        channel = normalize_channel_input(channel_raw)
        if not channel:
            st.error("Enter a channel URL or handle.")
        else:
            run_dir = create_run(channel, int(max_videos), int(holding_days), requested_mode)
            extraction_command = [
                sys.executable,
                str(PROJECT_ROOT / "run_pipeline.py"),
                "--channel",
                channel,
                "--max-videos",
                str(int(max_videos)),
                "--root",
                str(run_dir),
            ]
            if cookie_choice != "None":
                extraction_command.extend(["--cookies-from-browser", cookie_choice])

            extraction_code, extraction_log = run_step(
                extraction_command,
                "Transcripts and candidates",
                0.0,
                0.5,
                run_dir / "logs/extraction.log",
            )
            stats = run_stats(run_dir)
            if extraction_code != 0:
                update_manifest(
                    run_dir,
                    mode="failed",
                    message="Transcript extraction process failed",
                    error=extraction_log[-2000:],
                    stats=stats,
                )
            elif stats["transcripts"] == 0:
                update_manifest(
                    run_dir,
                    mode="diagnostic",
                    message="No transcripts were downloaded",
                    stats=stats,
                )
            elif stats["candidates"] == 0:
                update_manifest(
                    run_dir,
                    mode="diagnostic",
                    message="Transcripts downloaded, but no asset candidates were extracted",
                    stats=stats,
                )
            else:
                strict_command = [
                    sys.executable,
                    str(PROJECT_ROOT / "run_full_research_pipeline.py"),
                    "--channel",
                    channel,
                    "--max-videos",
                    str(int(max_videos)),
                    "--holding-days",
                    str(int(holding_days)),
                    "--root",
                    str(run_dir),
                ]
                if refresh_prices:
                    strict_command.append("--refresh-prices")
                strict_code, strict_log = run_step(
                    strict_command,
                    "Strict portfolio analysis",
                    0.5,
                    0.8,
                    run_dir / "logs/strict_analysis.log",
                )
                stats = run_stats(run_dir)
                mode = "strict"

                if stats["portfolio_days"] == 0 and requested_mode == "Loose exploration fallback":
                    loose_command = [
                        sys.executable,
                        str(PROJECT_ROOT / "loose_research_pipeline.py"),
                        "--root",
                        str(run_dir),
                        "--holding-days",
                        str(int(holding_days)),
                    ]
                    if refresh_prices:
                        loose_command.append("--refresh-prices")
                    run_step(
                        loose_command,
                        "Loose event analysis",
                        0.8,
                        1.0,
                        run_dir / "logs/loose_analysis.log",
                    )
                    stats = run_stats(run_dir)
                    mode = "loose" if stats["portfolio_days"] else "diagnostic"

                ok = stats["portfolio_days"] > 0
                if ok:
                    message = "Analysis completed"
                elif stats["events"] > 0:
                    message = "Source events were found, but no tradeable portfolio could be built"
                else:
                    message = "Candidates found, but no event dataset could be built"
                update_manifest(
                    run_dir,
                    mode=mode,
                    ok=ok,
                    message=message,
                    strict_exit_code=strict_code,
                    strict_error=strict_log[-2000:] if strict_code else "",
                    stats=stats,
                )

            st.cache_data.clear()
            st.rerun()


runs = discover_runs()
selected_run = None
with st.sidebar:
    st.header("Saved runs")
    if runs:
        labels = [run_label(run) for run in runs]
        selected_label = st.selectbox("Run", labels, index=0)
        selected_run = runs[labels.index(selected_label)]
        st.caption(read_manifest(selected_run).get("channel", ""))
    else:
        st.caption("No saved runs yet.")

if selected_run is None:
    st.info("Run a channel analysis to create the first saved result.")
    st.stop()

manifest = read_manifest(selected_run)
data = load_run_data(selected_run)
st.divider()
st.subheader("Run status")
if manifest.get("ok"):
    st.success(manifest.get("message", "Analysis completed"))
else:
    st.warning(manifest.get("message", "Diagnostic run"))
show_diagnostics(selected_run, data)

events = data["events"]
candidates = data["candidates"]
forward = data["forward"]
daily = data["daily"]
metrics = data["metrics"]
prices = data["prices"]

if events.empty:
    st.subheader("Source candidates")
    if candidates.empty:
        st.info("No source candidates are available for this run. The diagnostics above show where it stopped.")
    else:
        st.dataframe(
            source_table(candidates),
            hide_index=True,
            width="stretch",
            height=480,
            column_config={"timestamp_url": st.column_config.LinkColumn("Source timestamp")},
        )
    st.stop()

events["video_date"] = pd.to_datetime(events["video_date"], errors="coerce")
events = events.dropna(subset=["video_date"])
start = events["video_date"].min()
end = events["video_date"].max()

summary_columns = st.columns(4)
summary_columns[0].metric("Events", len(events))
summary_columns[1].metric("Unique tickers", events["ticker"].nunique())
summary_columns[2].metric("Sample start", start.date().isoformat() if pd.notna(start) else "")
summary_columns[3].metric("Sample end", end.date().isoformat() if pd.notna(end) else "")

performance_columns = st.columns(4)
performance_columns[0].metric(
    "Portfolio return",
    percent(metric_value(metrics, "Event portfolio", "total_return")),
)
performance_columns[1].metric(
    "Portfolio Sharpe",
    number(metric_value(metrics, "Event portfolio", "sharpe_0rf")),
)
performance_columns[2].metric(
    "Max drawdown",
    percent(metric_value(metrics, "Event portfolio", "max_drawdown")),
)
performance_columns[3].metric(
    "SPY return",
    percent(metric_value(metrics, "SPY", "total_return")),
)

if daily.empty:
    st.warning("Source-backed events exist, but this run did not produce a tradeable portfolio.")
else:
    st.plotly_chart(growth_chart(daily, prices, events), width="stretch")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(drawdown_chart(daily), width="stretch")
    with right:
        formatted_metrics = metrics.copy()
        for column in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
            if column in formatted_metrics.columns:
                formatted_metrics[column] = formatted_metrics[column].map(percent)
        if "sharpe_0rf" in formatted_metrics.columns:
            formatted_metrics["sharpe_0rf"] = formatted_metrics["sharpe_0rf"].map(number)
        st.dataframe(formatted_metrics, hide_index=True, width="stretch")

st.plotly_chart(event_timeline(events), width="stretch")
if not forward.empty:
    st.plotly_chart(forward_return_chart(forward), width="stretch")

available_tickers = sorted(set(events["ticker"]).intersection(prices.columns))
if available_tickers:
    selected_ticker = st.selectbox("Ticker price chart", available_tickers)
    st.plotly_chart(ticker_chart(events, prices, selected_ticker), width="stretch")

st.subheader("Source-backed events")
search = st.text_input("Search ticker, company, title or quote")
filtered = events.copy()
if search:
    query = re.escape(search.strip())
    searchable = [
        column
        for column in ["ticker", "company", "video_title", "quote_segment", "context_window"]
        if column in filtered.columns
    ]
    mask = pd.Series(False, index=filtered.index)
    for column in searchable:
        mask |= filtered[column].astype(str).str.contains(query, case=False, na=False)
    filtered = filtered[mask]

st.dataframe(
    source_table(filtered),
    hide_index=True,
    width="stretch",
    height=500,
    column_config={"timestamp_url": st.column_config.LinkColumn("Source timestamp")},
)
st.caption("Candidate rows must be manually verified before they are used in the final academic portfolio.")
