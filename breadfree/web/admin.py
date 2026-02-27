"""
BreadFree 管理控制台

功能:
  1. 行情数据 — ETF/个股行情查询、K线图、技术指标
  2. 新闻数据 — 新闻抓取、浏览、管理
  3. 回测中心 — 策略回测触发、历史回测记录
  4. 今日决策 — Top3 策略最新配置建议

启动:
    uv run streamlit run breadfree/web/admin.py
"""
import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

st.set_page_config(
    page_title="BreadFree 管理控制台",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from breadfree.web.pages import page_market, page_news, page_backtest, page_decision

# ── Sidebar ──
st.sidebar.title("🏦 BreadFree")
st.sidebar.caption("量化研究管理控制台")
st.sidebar.markdown("---")

page = st.sidebar.radio("导航", [
    "📊 行情数据",
    "📰 新闻数据",
    "🔬 回测中心",
    "🎯 今日决策",
])

st.sidebar.markdown("---")
from datetime import datetime
st.sidebar.caption(f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Router ──
PAGE_MAP = {
    "📊 行情数据": page_market.render,
    "📰 新闻数据": page_news.render,
    "🔬 回测中心": page_backtest.render,
    "🎯 今日决策": page_decision.render,
}

PAGE_MAP.get(page, page_market.render)()
