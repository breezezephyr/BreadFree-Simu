"""行情数据查询管理页"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from breadfree.web.page_modules.utils import get_pool, sym_name


def render():
    st.header("📊 行情数据查询")

    pool = get_pool()
    symbols = list(pool.keys())
    labels = [sym_name(s) for s in symbols]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_labels = st.multiselect("选择标的", labels, default=labels[:3])
    with col2:
        start = st.date_input("开始日期", datetime.now() - timedelta(days=90))
    with col3:
        end = st.date_input("结束日期", datetime.now())

    selected_codes = [s for s, l in zip(symbols, labels) if l in selected_labels]

    if not selected_codes:
        st.info("请选择至少一个标的")
        return

    if st.button("🔍 查询行情", type="primary"):
        _fetch_and_display(selected_codes, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))


def _fetch_and_display(codes, start_date, end_date):
    from breadfree.data.data_fetcher import DataFetcher
    fetcher = DataFetcher(data_dir="breadfree/data/cache", data_source="akshare")

    tabs = st.tabs([sym_name(c) for c in codes])
    for tab, code in zip(tabs, codes):
        with tab:
            with st.spinner(f"加载 {sym_name(code)} 数据..."):
                df = fetcher.fetch_a_stock_daily(code, start_date, end_date)

            if df.empty:
                st.warning(f"{sym_name(code)} 无数据")
                continue

            _render_symbol(code, df)


def _render_symbol(code: str, df: pd.DataFrame):
    name = sym_name(code)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 指标卡片
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    pct = (latest["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新价", f"{latest['close']:.3f}", f"{pct:+.2f}%")
    c2.metric("最高", f"{latest['high']:.3f}")
    c3.metric("最低", f"{latest['low']:.3f}")
    c4.metric("成交量", f"{latest.get('volume', 0):,.0f}")
    c5.metric("区间涨跌", f"{(df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100:+.2f}%")

    # K线走势
    chart_df = df[["close"]].copy()
    chart_df.columns = [name]
    st.line_chart(chart_df, height=300)

    # 均线对比
    if len(df) >= 20:
        ma_df = pd.DataFrame(index=df.index)
        ma_df["收盘价"] = df["close"]
        ma_df["MA5"] = df["close"].rolling(5).mean()
        ma_df["MA20"] = df["close"].rolling(20).mean()
        st.line_chart(ma_df.dropna(), height=250)

    # 原始数据表
    with st.expander("📋 查看原始数据"):
        display_df = df[["open", "close", "high", "low", "volume"]].copy()
        display_df.columns = ["开盘", "收盘", "最高", "最低", "成交量"]
        display_df = display_df.sort_index(ascending=False)
        st.dataframe(display_df, use_container_width=True, height=300)
