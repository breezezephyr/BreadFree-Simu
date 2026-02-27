"""Top3 策略今日决策报告"""
import streamlit as st
import pandas as pd
import json, os, time
from datetime import datetime, timedelta
from .utils import load_config, get_pool, sym_name


def render():
    st.header("🎯 Top3 策略今日决策报告")

    today = datetime.now()
    st.caption(f"报告日期: {today.strftime('%Y-%m-%d')} | 标的池: {len(get_pool())} 只")

    col1, col2 = st.columns([1, 1])
    with col1:
        lookback_days = st.slider("回测窗口（天）", 30, 365, 90)
    with col2:
        hold_period = st.slider("调仓周期", 5, 30, 20)

    if st.button("🚀 生成今日决策报告", type="primary"):
        _generate_report(lookback_days, hold_period)

    # 显示最近一次报告
    _show_latest_report()


def _generate_report(lookback_days, hold_period):
    from breadfree.engine.backtest_engine import BacktestEngine
    from breadfree.strategies.effi_rotation_strategy import RotationStrategy
    from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
    from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy
    from breadfree.utils.metrics import (
        calculate_total_return, calculate_max_drawdown,
        calculate_sharpe_ratio, calculate_annualized_return,
        calculate_win_rate, calculate_calmar_ratio,
    )

    pool = get_pool()
    symbols = list(pool.keys())
    today = datetime.now()
    sd = (today - timedelta(days=lookback_days)).strftime("%Y%m%d")
    ed = today.strftime("%Y%m%d")
    cash = load_config().get("initial_cash", 100000.0)

    strategies = [
        ("Rotation效率轮动", RotationStrategy,
         {"lookback_period": 20, "hold_period": hold_period, "top_n": 3, "use_efficiency": True}),
        ("EffiAgent-LLM效率轮动", EffiAgentRotationStrategy,
         {"lookback_period": 20, "hold_period": hold_period}),
        ("AgentV2-Bull/Bear辩证", AgentStrategyV2,
         {"lookback_period": 20, "hold_period": hold_period, "top_n": 3}),
    ]

    results = []
    progress = st.progress(0)

    for i, (name, cls, kwargs) in enumerate(strategies):
        progress.progress((i) / len(strategies), text=f"运行 {name}...")

        try:
            engine = BacktestEngine(
                strategy_cls=cls, symbols=symbols,
                start_date=sd, end_date=ed,
                initial_cash=cash, asset_type="stock",
                lot_size=100, data_source="akshare", **kwargs,
            )
            t0 = time.time()
            engine.run()
            elapsed = time.time() - t0

            if not engine.broker.equity_curve:
                results.append({"name": name, "error": "无数据"})
                continue

            es = pd.Series([d["equity"] for d in engine.broker.equity_curve])
            tr = [t["return_pct"] for t in engine.broker.closed_trades]
            fe = engine.broker.equity_curve[-1]["equity"]
            ret = calculate_total_return(es, initial_capital=cash)
            sh = calculate_sharpe_ratio(es)
            mdd = calculate_max_drawdown(es)
            wr, wc, tt = calculate_win_rate(tr)

            positions = {}
            for sym, pos in engine.broker.positions.items():
                qty = getattr(pos, 'quantity', pos)
                avg = getattr(pos, 'avg_price', 0)
                positions[sym] = {"qty": qty, "avg_price": avg}

            recent_tx = []
            for tx in engine.broker.transaction_history[-10:]:
                recent_tx.append({
                    "date": str(tx["date"])[:10],
                    "action": tx["action"],
                    "symbol": tx["symbol"],
                    "qty": tx["quantity"],
                    "price": tx["price"],
                })

            results.append({
                "name": name,
                "final_equity": fe,
                "total_return": ret,
                "sharpe": sh,
                "max_dd": mdd,
                "win_rate": wr,
                "trades": tt,
                "elapsed": elapsed,
                "positions": positions,
                "recent_tx": recent_tx,
                "cash": engine.broker.cash,
                "equity_curve": [{"date": str(d["date"])[:10], "equity": d["equity"]}
                                 for d in engine.broker.equity_curve],
            })
        except Exception as e:
            results.append({"name": name, "error": str(e)})

    progress.progress(1.0, text="完成!")

    # 保存报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "period": f"{sd}~{ed}",
        "initial_cash": cash,
        "results": results,
    }
    _save_report(report)
    _display_report(report)


def _display_report(report):
    results = report.get("results", [])
    if not results:
        st.warning("无结果")
        return

    st.subheader("📊 策略对比")

    # 汇总表
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"策略": r["name"], "状态": f"❌ {r['error']}"})
            continue
        rows.append({
            "策略": r["name"],
            "净值": f"¥{r['final_equity']:,.2f}",
            "总收益": f"{r['total_return']:.2%}",
            "Sharpe": f"{r['sharpe']:.2f}",
            "最大回撤": f"{r['max_dd']:.2%}",
            "胜率": f"{r['win_rate']:.0%}",
            "交易": r["trades"],
            "耗时": f"{r['elapsed']:.1f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 净值曲线叠加
    st.subheader("📈 净值曲线")
    chart_data = {}
    for r in results:
        if "error" in r:
            continue
        curve = r.get("equity_curve", [])
        if curve:
            for pt in curve:
                chart_data.setdefault(pt["date"], {})[r["name"]] = pt["equity"]
    if chart_data:
        chart_df = pd.DataFrame.from_dict(chart_data, orient="index").sort_index()
        st.line_chart(chart_df, height=350)

    # 各策略持仓详情
    st.subheader("📦 当前持仓建议")
    for r in results:
        if "error" in r:
            continue
        with st.expander(f"**{r['name']}** — 收益 {r['total_return']:.2%} | Sharpe {r['sharpe']:.2f}"):
            positions = r.get("positions", {})
            cash_val = r.get("cash", 0)
            fe = r.get("final_equity", 1)

            if positions:
                pos_rows = []
                for sym, info in sorted(positions.items(), key=lambda x: -(x[1]["qty"] * x[1]["avg_price"])):
                    val = info["qty"] * info["avg_price"]
                    pos_rows.append({
                        "标的": sym_name(sym),
                        "持仓": f"{info['qty']:,}股",
                        "均价": f"{info['avg_price']:.3f}",
                        "市值": f"¥{val:,.2f}",
                        "占比": f"{val / fe * 100:.1f}%",
                    })
                pos_rows.append({
                    "标的": "💰 现金",
                    "持仓": "",
                    "均价": "",
                    "市值": f"¥{cash_val:,.2f}",
                    "占比": f"{cash_val / fe * 100:.1f}%",
                })
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
            else:
                st.info("空仓（全部现金）")

            recent = r.get("recent_tx", [])
            if recent:
                st.caption("最近交易:")
                tx_rows = []
                for tx in recent:
                    tx_rows.append({
                        "日期": tx["date"],
                        "操作": tx["action"],
                        "标的": sym_name(tx["symbol"]),
                        "数量": f"{tx['qty']:,}股",
                        "价格": f"{tx['price']:.3f}",
                    })
                st.dataframe(pd.DataFrame(tx_rows), use_container_width=True, hide_index=True)


def _save_report(report):
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "latest_decision_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _show_latest_report():
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
    path = os.path.join(cache_dir, "latest_decision_report.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        gen_at = report.get("generated_at", "")[:19]
        st.info(f"📋 上次报告: {gen_at} | 区间: {report.get('period', '')}")

        with st.expander("查看上次报告详情"):
            _display_report(report)
    except Exception:
        pass
