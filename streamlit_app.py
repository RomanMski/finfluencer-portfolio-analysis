from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MARKET = ROOT / "data" / "market"

EVENTS_PATH = PROCESSED / "clean_candidate_events.csv"
FWD_PATH = PROCESSED / "forward_returns.csv"
DAILY_PATH = PROCESSED / "event_portfolio_daily.csv"
METRICS_PATH = PROCESSED / "performance_metrics.csv"
PRICES_PATH = MARKET / "adj_close.csv"


st.set_page_config(
    page_title="Finfluencer Portfolio Analysis",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=date_cols or [])


@st.cache_data(show_spinner=False)
def load_data():
    events = load_csv(EVENTS_PATH, ["video_date"])
    fwd = load_csv(FWD_PATH, ["video_date"]) if FWD_PATH.exists() else pd.DataFrame()
    daily = load_csv(DAILY_PATH, ["date"]) if DAILY_PATH.exists() else pd.DataFrame()
    metrics = load_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    prices = load_csv(PRICES_PATH, ["Date"]) if PRICES_PATH.exists() else pd.DataFrame()

    if not daily.empty and "date" in daily.columns:
        daily = daily.set_index("date").sort_index()

    if not prices.empty:
        first_col = prices.columns[0]
        if first_col.lower() in {"date", "index"}:
            prices = prices.set_index(first_col)
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index()

    return events, fwd, daily, metrics, prices


def pct(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.1%}"
    except Exception:
        return str(x)


def make_growth_chart(daily: pd.DataFrame):
    fig = go.Figure()
    series = [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]
    for col, label in series:
        if col in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=label))
    fig.update_layout(
        title="Demo event portfolio vs market benchmarks",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        height=480,
    )
    return fig


def make_drawdown_chart(daily: pd.DataFrame):
    fig = go.Figure()
    series = [
        ("event_portfolio_growth", "Event portfolio"),
        ("SPY_growth", "SPY"),
        ("QQQ_growth", "QQQ"),
    ]
    for col, label in series:
        if col in daily.columns:
            g = daily[col].dropna()
            dd = g / g.cummax() - 1
            fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=label))
    fig.update_layout(
        title="Drawdown comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def make_ticker_chart(events: pd.DataFrame, prices: pd.DataFrame, ticker: str):
    fig = go.Figure()

    if prices.empty or ticker not in prices.columns:
        fig.update_layout(
            title=f"{ticker}: price data not available",
            template="plotly_white",
            height=520,
        )
        return fig

    px = prices[[ticker]].dropna().copy()
    px = px.rename(columns={ticker: "price"})

    ticker_events = events[events["ticker"].eq(ticker)].copy()
    if not ticker_events.empty:
        start = ticker_events["video_date"].min() - pd.Timedelta(days=20)
        end = ticker_events["video_date"].max() + pd.Timedelta(days=120)
        px = px[(px.index >= start) & (px.index <= end)]

    fig.add_trace(
        go.Scatter(
            x=px.index,
            y=px["price"],
            mode="lines",
            name=f"{ticker} adjusted close",
            line=dict(width=2),
        )
    )

    marker_specs = {
        "buy_or_add": ("triangle-up", "Buy/add candidate"),
        "sell_or_reduce": ("triangle-down", "Sell/reduce candidate"),
        "holding_update": ("circle", "Holding update"),
    }

    for event_type, (symbol, label) in marker_specs.items():
        ev = ticker_events[ticker_events["event_type"].eq(event_type)].copy()
        if ev.empty:
            continue

        marker_x = []
        marker_y = []
        marker_text = []

        for _, row in ev.iterrows():
            d = pd.Timestamp(row["video_date"])
            idx = px.index.searchsorted(d, side="left")
            if idx >= len(px.index):
                continue
            actual_d = px.index[idx]
            marker_x.append(actual_d)
            marker_y.append(px.loc[actual_d, "price"])

            title = str(row.get("video_title", ""))[:90]
            ts_url = str(row.get("timestamp_url", ""))
            quote = str(row.get("quote_segment", ""))[:180]
            marker_text.append(
                f"<b>{event_type}</b><br>{actual_d.date()}<br>{title}<br>{quote}<br>{ts_url}"
            )

        if marker_x:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    name=label,
                    marker=dict(size=13, symbol=symbol),
                    text=marker_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{ticker}: adjusted price with extracted recommendation markers",
        xaxis_title="Date",
        yaxis_title="Adjusted close",
        hovermode="closest",
        template="plotly_white",
        height=560,
    )
    return fig


def make_timeline(events: pd.DataFrame):
    if events.empty:
        return go.Figure()

    top = events["ticker"].value_counts().head(20).index
    df = events[events["ticker"].isin(top)].copy()
    df["video_date"] = pd.to_datetime(df["video_date"], errors="coerce")

    fig = px.scatter(
        df,
        x="video_date",
        y="ticker",
        color="event_type",
        hover_data=["video_title", "timestamp_url", "quote_segment"],
        title="Extracted recommendation timeline",
        template="plotly_white",
        height=560,
    )
    fig.update_layout(xaxis_title="Video upload date", yaxis_title="Ticker")
    return fig


events, fwd, daily, metrics, prices = load_data()

st.title("Finfluencer Portfolio Analysis")
st.caption(
    "Transcript-to-dataset-to-performance dashboard. The current data are first-pass candidates and need manual verification before final academic use."
)

if events.empty:
    st.error(
        "No cleaned event data found. Run `python run_full_research_pipeline.py ...` first."
    )
    st.stop()

min_date = pd.to_datetime(events["video_date"]).min()
max_date = pd.to_datetime(events["video_date"]).max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidate events", f"{len(events):,}")
col2.metric("Unique tickers", f"{events['ticker'].nunique():,}")
col3.metric("Sample start", min_date.date().isoformat() if pd.notna(min_date) else "")
col4.metric("Sample end", max_date.date().isoformat() if pd.notna(max_date) else "")

st.divider()

with st.sidebar:
    st.header("Filters")
    tickers = sorted(events["ticker"].dropna().unique().tolist())
    default_ticker = tickers[0] if tickers else None
    # Prefer popular/interesting tickers if present.
    for preferred in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]:
        if preferred in tickers:
            default_ticker = preferred
            break

    selected_ticker = st.selectbox(
        "Ticker for price/event chart",
        tickers,
        index=tickers.index(default_ticker) if default_ticker in tickers else 0,
    )

    selected_actions = st.multiselect(
        "Event types shown in table",
        sorted(events["event_type"].dropna().unique().tolist()),
        default=sorted(events["event_type"].dropna().unique().tolist()),
    )

    st.caption(
        "For final report work, open timestamp links and verify the rows used in the portfolio."
    )


tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Ticker event chart", "Forward returns", "Source table"]
)

with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        if not daily.empty:
            st.plotly_chart(make_growth_chart(daily), use_container_width=True)
        else:
            st.warning("Daily portfolio file not found.")

    with right:
        if not metrics.empty:
            show = metrics.copy()
            for c in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
                if c in show.columns:
                    show[c] = show[c].map(pct)
            if "sharpe_0rf" in show.columns:
                show["sharpe_0rf"] = show["sharpe_0rf"].map(
                    lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
                )
            st.subheader("Performance metrics")
            st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.warning("Performance metrics file not found.")

        counts = events["event_type"].value_counts().reset_index()
        counts.columns = ["event_type", "count"]
        fig_counts = px.bar(
            counts,
            x="count",
            y="event_type",
            orientation="h",
            title="Candidate event types",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig_counts, use_container_width=True)

    if not daily.empty:
        st.plotly_chart(make_drawdown_chart(daily), use_container_width=True)

    st.plotly_chart(make_timeline(events), use_container_width=True)

with tab2:
    st.subheader(f"{selected_ticker}: price chart with extracted events")
    st.plotly_chart(make_ticker_chart(events, prices, selected_ticker), use_container_width=True)

    tdf = events[events["ticker"].eq(selected_ticker)].sort_values("video_date").copy()
    st.dataframe(
        tdf[
            [
                "video_date",
                "event_type",
                "company",
                "video_title",
                "timestamp_url",
                "quote_segment",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

with tab3:
    st.subheader("Forward returns after extracted events")
    if fwd.empty:
        st.warning("Forward returns file not found.")
    else:
        horizon_cols = [c for c in fwd.columns if c.startswith("return_")]
        if horizon_cols:
            long = []
            for c in horizon_cols:
                tmp = fwd[["ticker", "event_type", c]].copy()
                tmp["horizon"] = c.replace("return_", "")
                tmp["return"] = tmp[c]
                long.append(tmp[["ticker", "event_type", "horizon", "return"]])
            fr = pd.concat(long, ignore_index=True).dropna()

            avg = (
                fr.groupby(["horizon", "event_type"])["return"]
                .mean()
                .reset_index()
            )
            fig = px.bar(
                avg,
                x="horizon",
                y="return",
                color="event_type",
                barmode="group",
                title="Average forward return by event type",
                template="plotly_white",
                height=450,
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            top_table = (
                fr[fr["horizon"].eq("3m")]
                .sort_values("return", ascending=False)
                .head(20)
            )
            st.subheader("Top 3-month forward returns")
            st.dataframe(top_table, hide_index=True, use_container_width=True)
        else:
            st.warning("No return_* columns found.")

with tab4:
    st.subheader("Timestamped source table")
    t = events.copy()
    if selected_actions:
        t = t[t["event_type"].isin(selected_actions)]

    search = st.text_input("Search title / quote / ticker")
    if search:
        s = search.lower()
        mask = (
            t["ticker"].astype(str).str.lower().str.contains(s, na=False)
            | t["video_title"].astype(str).str.lower().str.contains(s, na=False)
            | t["quote_segment"].astype(str).str.lower().str.contains(s, na=False)
            | t["context_window"].astype(str).str.lower().str.contains(s, na=False)
        )
        t = t[mask]

    cols = [
        "video_date",
        "ticker",
        "event_type",
        "company",
        "video_title",
        "timestamp_url",
        "quote_segment",
        "context_window",
    ]
    cols = [c for c in cols if c in t.columns]
    st.dataframe(t[cols].sort_values(["video_date", "ticker"]), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "This dashboard is for exploration and presentation. The final project should use manually verified recommendation rows and clearly documented portfolio rules."
)
