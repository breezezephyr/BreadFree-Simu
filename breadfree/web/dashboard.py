"""
BreadFree Live Trading Dashboard (Streamlit)

A real-time monitoring panel for the live trading engine.

Pages:
1. Account Overview  - equity, cash, P&L
2. Positions         - current holdings with unrealized P&L
3. Order & Trade Log - today's orders and fills
4. Equity Curve      - historical equity chart
5. Audit Trail       - strategy decisions, risk events, LLM calls
6. Emergency Control - pause strategy, manual commands

Run:
    uv run streamlit run breadfree/web/dashboard.py
"""

import sys
import os
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd

from breadfree.data.live_store import LiveTradeStore
from breadfree.monitor.audit_logger import AuditLogger


# ──────────────────────────────────────────
# Page config
# ──────────────────────────────────────────

st.set_page_config(
    page_title="BreadFree Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────
# Initialize data sources
# ──────────────────────────────────────────

@st.cache_resource
def get_store():
    db_path = os.environ.get("BREADFREE_LIVE_DB", "live_trading.db")
    return LiveTradeStore(db_path)


@st.cache_resource
def get_audit():
    db_path = os.environ.get("BREADFREE_LIVE_DB", "live_trading.db")
    return AuditLogger(db_path)


store = get_store()
audit = get_audit()


# ──────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────

st.sidebar.title("BreadFree")
st.sidebar.caption("Live Trading Dashboard")

page = st.sidebar.radio("Navigation", [
    "Account Overview",
    "Positions",
    "Orders & Trades",
    "Equity Curve",
    "Audit Trail",
    "System",
])

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# ──────────────────────────────────────────
# Page: Account Overview
# ──────────────────────────────────────────

def page_account_overview():
    st.header("Account Overview")

    summary = store.get_summary()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        equity = summary.get("latest_equity")
        st.metric("Total Equity", f"¥{equity:,.2f}" if equity else "N/A")
    with col2:
        st.metric("Total Orders", summary.get("total_orders", 0))
    with col3:
        st.metric("Total Trades", summary.get("total_trades", 0))
    with col4:
        st.metric("Tracking Days", summary.get("equity_curve_points", 0))

    # Engine state
    st.subheader("Engine State")
    state = store.get_all_state()
    if state:
        state_df = pd.DataFrame(
            [{"Key": k, "Value": v} for k, v in state.items()]
        )
        st.dataframe(state_df, use_container_width=True, hide_index=True)
    else:
        st.info("No engine state recorded yet. Start the live engine to see data.")

    # Today's activity
    st.subheader("Today's Activity")
    today_orders = store.get_today_orders()
    today_trades = store.get_today_trades()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Orders Today", len(today_orders))
    with col2:
        st.metric("Trades Today", len(today_trades))


# ──────────────────────────────────────────
# Page: Positions
# ──────────────────────────────────────────

def page_positions():
    st.header("Current Positions")

    # Get latest position snapshot
    positions = store.get_latest_position_snapshot()

    if not positions:
        st.info("No position snapshot available.")
        return

    df = pd.DataFrame(positions)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Summary metrics
    total_value = df["market_value"].sum() if "market_value" in df.columns else 0
    total_pnl = df["unrealized_pnl"].sum() if "unrealized_pnl" in df.columns else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Holdings", len(df))
    with col2:
        st.metric("Total Market Value", f"¥{total_value:,.2f}")
    with col3:
        delta_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Unrealized P&L", f"¥{total_pnl:,.2f}",
                  delta=f"{total_pnl:+,.2f}", delta_color=delta_color)

    # Position table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_value": st.column_config.NumberColumn("Market Value", format="¥%.2f"),
            "avg_price": st.column_config.NumberColumn("Avg Price", format="%.4f"),
            "market_price": st.column_config.NumberColumn("Mkt Price", format="%.4f"),
            "unrealized_pnl": st.column_config.NumberColumn("Unrealized P&L", format="¥%.2f"),
        },
    )


# ──────────────────────────────────────────
# Page: Orders & Trades
# ──────────────────────────────────────────

def page_orders_trades():
    st.header("Orders & Trades")

    tab1, tab2 = st.tabs(["Recent Orders", "Recent Trades"])

    with tab1:
        orders = store.get_recent_orders(limit=100)
        if orders:
            df = pd.DataFrame(orders)
            # Color code status
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No orders recorded yet.")

    with tab2:
        trades = store.get_today_trades()
        if not trades:
            # Fall back to recent trades from store
            # (get_today_trades may be empty if no trading today)
            trades = []
        if trades:
            df = pd.DataFrame(trades)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")


# ──────────────────────────────────────────
# Page: Equity Curve
# ──────────────────────────────────────────

def page_equity_curve():
    st.header("Equity Curve")

    curve = store.get_equity_curve()

    if not curve:
        st.info("No equity curve data. The live engine saves a point each day after market close.")
        return

    df = pd.DataFrame(curve)
    df["record_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values("record_date")

    # Summary
    if len(df) >= 2:
        start_eq = df.iloc[0]["total_equity"]
        end_eq = df.iloc[-1]["total_equity"]
        total_return = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
        peak = df["total_equity"].max()
        drawdown = (peak - end_eq) / peak * 100 if peak > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Equity", f"¥{end_eq:,.2f}")
        with col2:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col3:
            st.metric("Current Drawdown", f"{drawdown:.2f}%")

    # Chart
    st.line_chart(df.set_index("record_date")[["total_equity"]])

    # Breakdown chart
    if "available_cash" in df.columns and "position_value" in df.columns:
        st.subheader("Equity Breakdown")
        chart_df = df.set_index("record_date")[["available_cash", "position_value"]]
        st.area_chart(chart_df)


# ──────────────────────────────────────────
# Page: Audit Trail
# ──────────────────────────────────────────

def page_audit_trail():
    st.header("Audit Trail")

    # Category filter
    col1, col2 = st.columns([1, 3])
    with col1:
        category = st.selectbox("Category", [
            "All", "ORDER", "TRADE", "RISK", "STRATEGY", "LLM", "SYSTEM", "GATEWAY"
        ])
    with col2:
        limit = st.slider("Max entries", 20, 500, 100)

    cat_filter = None if category == "All" else category
    entries = audit.get_recent(limit=limit, category=cat_filter)

    if not entries:
        st.info("No audit entries yet.")
        return

    df = pd.DataFrame(entries)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Summary counts
    if "category" in df.columns:
        counts = df["category"].value_counts()
        st.bar_chart(counts)

    # Table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Warnings / errors section
    st.subheader("Warnings & Errors")
    issues = audit.get_warnings_and_errors(limit=20)
    if issues:
        issue_df = pd.DataFrame(issues)
        if "id" in issue_df.columns:
            issue_df = issue_df.drop(columns=["id"])
        st.dataframe(issue_df, use_container_width=True, hide_index=True)
    else:
        st.success("No warnings or errors recorded.")


# ──────────────────────────────────────────
# Page: System
# ──────────────────────────────────────────

def page_system():
    st.header("System Information")

    st.subheader("Database Summary")
    summary = store.get_summary()
    for key, val in summary.items():
        st.text(f"  {key}: {val}")

    st.subheader("Audit Category Counts (Today)")
    counts = audit.count_by_category()
    if counts:
        df = pd.DataFrame([{"Category": k, "Count": v} for k, v in counts.items()])
        st.bar_chart(df.set_index("Category"))
    else:
        st.info("No audit entries today.")

    st.subheader("Configuration")
    st.json({
        "live_db": os.environ.get("BREADFREE_LIVE_DB", "live_trading.db"),
        "dashboard_version": "1.0.0",
    })


# ──────────────────────────────────────────
# Router
# ──────────────────────────────────────────

PAGE_MAP = {
    "Account Overview": page_account_overview,
    "Positions": page_positions,
    "Orders & Trades": page_orders_trades,
    "Equity Curve": page_equity_curve,
    "Audit Trail": page_audit_trail,
    "System": page_system,
}

PAGE_MAP.get(page, page_account_overview)()
