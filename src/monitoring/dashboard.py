"""Streamlit dashboard for monitoring the trading system."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

from src.config import load_config
from src.database import Database
from src.execution.broker import AlpacaBroker

st.set_page_config(page_title="Swing Trader Dashboard", layout="wide")

@st.cache_resource
def get_broker():
    config = load_config()
    return AlpacaBroker(config)


@st.cache_resource
def get_db():
    config = load_config()
    return Database(config)


def main():
    st.title("Swing Trader Dashboard")

    db = get_db()

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Overview", "Open Positions", "Trade History", "Equity Curve",
         "Signal Pipeline", "Decision Log", "Backtest Results"],
    )

    if page == "Overview":
        render_overview(db)
    elif page == "Open Positions":
        broker = get_broker()
        render_open_positions(db, broker)
    elif page == "Trade History":
        render_trade_history(db)
    elif page == "Equity Curve":
        render_equity_curve(db)
    elif page == "Signal Pipeline":
        render_signal_pipeline(db)
    elif page == "Decision Log":
        render_decision_log(db)
    elif page == "Backtest Results":
        render_backtest_results()


def render_overview(db: Database):
    """Overview page with key metrics."""
    st.header("System Overview")

    # Key metrics
    equity_curve = db.get_equity_curve()
    open_trades = db.get_open_trades()

    if equity_curve:
        latest = equity_curve[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Equity", f"${latest['total_equity']:,.2f}")
        col2.metric("Daily P&L", f"${latest['daily_pnl']:+,.2f}")
        col3.metric("Drawdown", f"{latest['drawdown_pct']:.2%}")
        col4.metric("Open Positions", latest["open_positions"])
    else:
        st.info("No equity data yet. Run the system or a backtest first.")

    # Recent trades
    st.subheader("Recent Trades")
    with db.connect() as conn:
        cursor = conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT 20"
        )
        trades = [dict(row) for row in cursor.fetchall()]

    if trades:
        df = pd.DataFrame(trades)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trades yet.")


def render_open_positions(db: Database, broker: AlpacaBroker = None):
    """Open positions page."""
    st.header("Open Positions")

    # Add Live Broker Positions
    if broker:
        st.subheader("Live Broker Positions (Alpaca)")
        try:
            live_positions = broker.get_positions()
            if live_positions:
                st.dataframe(pd.DataFrame(live_positions), use_container_width=True)
            else:
                st.info("No active positions on Alpaca.")
        except Exception as e:
            st.error(f"Failed to load live positions: {e}")

    st.subheader("Database Tracked Positions")
    trades = db.get_open_trades()
    if not trades:
        st.info("No open trades tracked in the local database.")
        return

    df = pd.DataFrame(trades)
    st.dataframe(df, use_container_width=True)

    # Position summary
    st.subheader("Position Breakdown")
    for trade in trades:
        with st.expander(f"{trade['ticker']} — {trade['direction'].upper()}"):
            col1, col2, col3 = st.columns(3)
            entry = f"${trade['entry_price']:.2f}" if trade['entry_price'] is not None else "Pending"
            stop = f"${trade['stop_loss_price']:.2f}" if trade['stop_loss_price'] is not None else "N/A"
            col1.write(f"**Entry:** {entry}")
            col2.write(f"**Stop Loss:** {stop}")
            col3.write(f"**Shares:** {trade['shares']}")
            st.write(f"Signal: {trade['signal_type']} (score: {trade['signal_score']:.2f})")
            if trade.get("entry_reason"):
                st.write(f"Reason: {trade['entry_reason']}")
            st.write(f"Target Exit: {trade['target_exit_date']}")


def render_trade_history(db: Database):
    """Historical trades page."""
    st.header("Trade History")

    with db.connect() as conn:
        cursor = conn.execute(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_date DESC"
        )
        trades = [dict(row) for row in cursor.fetchall()]

    if not trades:
        st.info("No closed trades yet.")
        return

    df = pd.DataFrame(trades)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", len(df))
    col2.metric("Win Rate", f"{(df['pnl'] > 0).mean():.1%}")
    col3.metric("Avg P&L", f"{df['pnl_pct'].mean():+.2%}")
    col4.metric("Total P&L", f"${df['pnl'].sum():+,.2f}")

    # P&L by signal type
    st.subheader("P&L by Signal Type")
    signal_stats = df.groupby("signal_type").agg(
        count=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl_pct=("pnl_pct", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()

    fig = px.bar(signal_stats, x="signal_type", y="total_pnl",
                 title="Total P&L by Signal Type", color="total_pnl",
                 color_continuous_scale=["red", "green"])
    st.plotly_chart(fig, use_container_width=True)

    # Full trade table
    st.subheader("All Trades")
    st.dataframe(df, use_container_width=True)


def render_equity_curve(db: Database):
    """Equity curve visualization."""
    st.header("Equity Curve")

    curve = db.get_equity_curve()
    if not curve:
        st.info("No equity data yet.")
        return

    df = pd.DataFrame(curve)
    df["date"] = pd.to_datetime(df["date"])

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_equity"],
        mode="lines", name="Portfolio",
        line=dict(color="#2196F3", width=2),
    ))
    fig.update_layout(title="Portfolio Equity", yaxis_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["date"], y=-df["drawdown_pct"] * 100,
        mode="lines", fill="tozeroy", name="Drawdown",
        line=dict(color="#F44336"),
    ))
    fig2.update_layout(title="Drawdown", yaxis_title="Drawdown (%)")
    st.plotly_chart(fig2, use_container_width=True)


def render_signal_pipeline(db: Database):
    """Signal pipeline: recent signals detected."""
    st.header("Signal Pipeline")

    with db.connect() as conn:
        cursor = conn.execute(
            "SELECT * FROM signals ORDER BY signal_date DESC, strength DESC LIMIT 100"
        )
        signals = [dict(row) for row in cursor.fetchall()]

    if not signals:
        st.info("No signals recorded yet.")
        return

    df = pd.DataFrame(signals)

    # Filter by signal type
    signal_types = df["signal_type"].unique().tolist()
    selected = st.multiselect("Filter by signal type", signal_types, default=signal_types)
    filtered = df[df["signal_type"].isin(selected)]

    st.dataframe(filtered, use_container_width=True)


def render_decision_log(db: Database):
    """Decision trace page: shows selected/rejected candidates with reasoning."""
    st.header("Decision Log")

    rows = db.get_recent_decision_logs(limit=500)
    if not rows:
        st.info("No decision logs yet. Run a scan/trade cycle first.")
        return

    df = pd.DataFrame(rows)
    df["selected"] = df["selected"].astype(bool)

    col1, col2, col3 = st.columns(3)
    mode_options = sorted(df["mode"].dropna().unique().tolist())
    stage_options = sorted(df["stage"].dropna().unique().tolist())
    selected_modes = col1.multiselect("Mode", mode_options, default=mode_options)
    selected_stages = col2.multiselect("Stage", stage_options, default=stage_options)
    selected_flag = col3.selectbox("Decision type", ["All", "Selected", "Rejected"])

    filtered = df[df["mode"].isin(selected_modes) & df["stage"].isin(selected_stages)]
    if selected_flag == "Selected":
        filtered = filtered[filtered["selected"]]
    elif selected_flag == "Rejected":
        filtered = filtered[~filtered["selected"]]

    st.subheader("Latest Decisions")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Reasoning Details")
    for row in filtered.head(50).to_dict(orient="records"):
        verdict = "SELECTED" if row["selected"] else "REJECTED"
        title = (
            f"{row['decision_time']} | {row['ticker']} | {verdict} | "
            f"mode={row['mode']} stage={row['stage']}"
        )
        with st.expander(title):
            st.write(f"Direction: {row.get('direction') or 'n/a'}")
            st.write(f"Score: {row.get('score')}")
            st.write(f"Signals: {row.get('signal_sources') or 'n/a'}")
            if row.get("reasoning"):
                st.write(f"Reasoning: {row['reasoning']}")
            if row.get("rejection_reason"):
                st.write(f"Rejection reason: {row['rejection_reason']}")
            signal_details_raw = row.get("signal_details")
            if signal_details_raw:
                try:
                    details = json.loads(signal_details_raw)
                except Exception:
                    details = []
                if details:
                    st.write("Signal details:")
                    for sd in details:
                        st.write(
                            f"- {sd.get('signal_type')} | dir={sd.get('direction')} | "
                            f"strength={sd.get('strength')} | confidence={sd.get('confidence')}"
                        )
                        meta = sd.get("metadata") or {}
                        if meta:
                            st.json(meta)
            agent_trace_raw = row.get("agent_trace")
            if agent_trace_raw:
                try:
                    agent_trace = json.loads(agent_trace_raw)
                except Exception:
                    agent_trace = {}
                if agent_trace:
                    st.write("Agent trace:")
                    st.json(agent_trace)


def render_backtest_results():
    """Load and display saved backtest results."""
    st.header("Backtest Results")

    results_dir = ROOT / "data" / "backtest_results"
    if not results_dir.exists():
        st.info("No backtest results found. Run a backtest first.")
        return

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        st.info("No backtest results found.")
        return

    selected_file = st.selectbox("Select result", [f.stem for f in json_files])
    result_path = results_dir / f"{selected_file}.json"

    import json
    with open(result_path) as f:
        data = json.load(f)

    # Summary
    summary = data.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{summary.get('total_return_pct', 0):+.2%}")
    col2.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
    col3.metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2%}")
    col4.metric("Win Rate", f"{summary.get('win_rate', 0):.1%}")

    # Equity chart
    chart_path = results_dir / f"{selected_file}_equity.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Equity Curve")

    dist_path = results_dir / f"{selected_file}_distribution.png"
    if dist_path.exists():
        st.image(str(dist_path), caption="Trade Distribution")

    # Trade table
    trades = data.get("trades", [])
    if trades:
        st.subheader(f"Trades ({len(trades)})")
        st.dataframe(pd.DataFrame(trades), use_container_width=True)


if __name__ == "__main__":
    main()
