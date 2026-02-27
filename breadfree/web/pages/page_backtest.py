"""回测中心 — 策略回测触发 + 历史记录管理"""
import streamlit as st
import pandas as pd
import json, os, time, subprocess
from datetime import datetime, timedelta
from .utils import load_config, get_pool, sym_name

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "backtest_history.json")


def render():
    st.header("🔬 回测中心")

    tab1, tab2 = st.tabs(["🚀 发起回测", "📋 历史记录"])

    with tab1:
        _render_backtest_form()
    with tab2:
        _render_history()


def _render_backtest_form():
    cfg = load_config()
    pool = get_pool()

    st.subheader("策略参数配置")

    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("策略", [
            "RotationStrategy", "BenchmarkStrategy", "DoubleMAStrategy",
            "TripleMomentumStrategy", "AgentStrategyV2", "EffiA",
        ])
    with col2:
        use_all = st.checkbox("使用全部标的池", value=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("开始日期", datetime(2025, 1, 1))
    with col2:
        end = st.date_input("结束日期", datetime.now())
    with col3:
        cash = st.number_input("初始资金", value=100000, step=10000)

    if strategy in ["RotationStrategy", "AgentStrategyV2", "EffiA"]:
        st.subheader("轮动参数")
        col1, col2, col3 = st.columns(3)
        with col1:
            lookback = st.slider("回看周期", 5, 60, 20)
        with col2:
            hold = st.slider("持仓周期", 5, 40, 20)
        with col3:
            top_n = st.slider("持仓数量", 1, 10, 3)
    else:
        lookback, hold, top_n = 20, 20, 3

    if st.button("▶️ 开始回测", type="primary"):
        _run_backtest(strategy, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                      cash, lookback, hold, top_n)


def _run_backtest(strategy, sd, ed, cash, lookback, hold, top_n):
    progress = st.progress(0, text="初始化回测引擎...")

    try:
        from breadfree.engine.backtest_engine import BacktestEngine
        from breadfree.strategies.effi_rotation_strategy import RotationStrategy
        from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
        from breadfree.strategies.ma_strategy import DoubleMAStrategy
        from breadfree.strategies.triple_momentum_strategy import TripleMomentumStrategy
        from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
        from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy
        from breadfree.utils.metrics import (
            calculate_total_return, calculate_max_drawdown,
            calculate_sharpe_ratio, calculate_annualized_return,
            calculate_profit_loss_ratio, calculate_win_rate, calculate_calmar_ratio,
        )

        strategy_map = {
            "RotationStrategy": RotationStrategy,
            "BenchmarkStrategy": BenchmarkStrategy,
            "DoubleMAStrategy": DoubleMAStrategy,
            "TripleMomentumStrategy": TripleMomentumStrategy,
            "AgentStrategyV2": AgentStrategyV2,
            "EffiA": EffiAgentRotationStrategy,
        }
        cls = strategy_map[strategy]
        pool = get_pool()
        symbols = list(pool.keys())

        kwargs = {}
        if strategy in ["RotationStrategy", "AgentStrategyV2", "EffiA"]:
            kwargs = {"lookback_period": lookback, "hold_period": hold, "top_n": top_n}
        if strategy == "RotationStrategy":
            kwargs["use_efficiency"] = True

        progress.progress(0.1, text="获取行情数据...")

        engine = BacktestEngine(
            strategy_cls=cls, symbols=symbols,
            start_date=sd, end_date=ed,
            initial_cash=cash, asset_type="stock",
            lot_size=100, data_source="akshare", **kwargs,
        )

        progress.progress(0.3, text="运行回测中...")
        t0 = time.time()
        engine.run()
        elapsed = time.time() - t0

        progress.progress(0.9, text="计算指标...")

        if not engine.broker.equity_curve:
            st.error("回测无数据")
            return

        es = pd.Series([d["equity"] for d in engine.broker.equity_curve])
        tr = [t["return_pct"] for t in engine.broker.closed_trades]
        fe = engine.broker.equity_curve[-1]["equity"]
        ret = calculate_total_return(es, initial_capital=cash)
        ann = calculate_annualized_return(es, annual_days=242)
        sh = calculate_sharpe_ratio(es)
        mdd = calculate_max_drawdown(es)
        cal = calculate_calmar_ratio(ann, mdd, risk_free_rate=0.015)
        wr, wc, tt = calculate_win_rate(tr)

        progress.progress(1.0, text="完成!")

        # 展示结果
        st.subheader("📈 回测结果")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("最终净值", f"¥{fe:,.2f}")
        c2.metric("总收益率", f"{ret:.2%}")
        c3.metric("Sharpe", f"{sh:.2f}")
        c4.metric("最大回撤", f"{mdd:.2%}")
        c5.metric("胜率", f"{wr:.0%}({wc}/{tt})")

        c1, c2, c3 = st.columns(3)
        c1.metric("年化收益", f"{ann:.2%}")
        c2.metric("Calmar", f"{cal:.2f}")
        c3.metric("耗时", f"{elapsed:.1f}s")

        # 净值曲线
        eq_df = pd.DataFrame(engine.broker.equity_curve)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        st.line_chart(eq_df.set_index("date")["equity"], height=300)

        # 持仓
        if engine.broker.positions:
            st.subheader("📦 最终持仓")
            pos_rows = []
            for sym, pos in engine.broker.positions.items():
                qty = getattr(pos, 'quantity', pos)
                avg = getattr(pos, 'avg_price', 0)
                pos_rows.append({"标的": sym_name(sym), "数量": qty, "均价": f"{avg:.3f}",
                                 "市值": f"¥{qty * avg:,.2f}"})
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        # 交易记录
        if engine.broker.transaction_history:
            st.subheader("📝 交易记录")
            tx_df = pd.DataFrame(engine.broker.transaction_history)
            tx_df["symbol"] = tx_df["symbol"].apply(sym_name)
            tx_df["date"] = tx_df["date"].apply(lambda x: str(x)[:10])
            st.dataframe(tx_df, use_container_width=True, hide_index=True, height=300)

        # 保存历史
        record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "start_date": sd, "end_date": ed,
            "initial_cash": cash,
            "params": {"lookback": lookback, "hold": hold, "top_n": top_n},
            "results": {
                "final_equity": round(fe, 2),
                "total_return": round(ret * 100, 2),
                "annualized": round(ann * 100, 2),
                "sharpe": round(sh, 2),
                "max_drawdown": round(mdd * 100, 2),
                "calmar": round(cal, 2),
                "win_rate": round(wr * 100, 1),
                "trades": tt,
                "elapsed_sec": round(elapsed, 1),
            },
        }
        _save_history(record)

    except Exception as e:
        st.error(f"回测失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def _save_history(record):
    history = _load_history()
    history.insert(0, record)
    history = history[:100]
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _render_history():
    history = _load_history()
    if not history:
        st.info("暂无历史回测记录，请先发起回测")
        return

    st.subheader(f"共 {len(history)} 条记录")

    rows = []
    for h in history:
        r = h.get("results", {})
        rows.append({
            "时间": h["timestamp"][:19],
            "策略": h["strategy"],
            "区间": f"{h['start_date']}~{h['end_date']}",
            "初始资金": f"¥{h['initial_cash']:,.0f}",
            "总收益": f"{r.get('total_return', 0):.1f}%",
            "年化": f"{r.get('annualized', 0):.1f}%",
            "Sharpe": r.get("sharpe", 0),
            "最大回撤": f"{r.get('max_drawdown', 0):.1f}%",
            "Calmar": r.get("calmar", 0),
            "胜率": f"{r.get('win_rate', 0):.0f}%",
            "交易": r.get("trades", 0),
            "耗时": f"{r.get('elapsed_sec', 0):.1f}s",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("🗑️ 清空历史记录"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("已清空")
        st.rerun()
