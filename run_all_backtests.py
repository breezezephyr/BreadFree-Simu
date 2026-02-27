#!/usr/bin/env python3
"""全策略回测分析 — 运行所有非LLM策略, 生成综合收益报告"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yaml

from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.ma_strategy import DoubleMAStrategy
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.strategies.triple_momentum_strategy import TripleMomentumStrategy
from breadfree.utils.metrics import (
    calculate_total_return, calculate_max_drawdown,
    calculate_sharpe_ratio, calculate_annualized_return,
    calculate_profit_loss_ratio, calculate_win_rate, calculate_calmar_ratio,
)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "breadfree", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_single_backtest(strategy_cls, symbols, start_date, end_date,
                        initial_cash, data_source, lot_size=100, **kwargs):
    engine = BacktestEngine(
        strategy_cls=strategy_cls,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        asset_type="stock",
        lot_size=lot_size,
        data_source=data_source,
        **kwargs,
    )
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0

    if not engine.broker.equity_curve:
        return None

    equity_series = pd.Series([d["equity"] for d in engine.broker.equity_curve])
    trade_returns = [t["return_pct"] for t in engine.broker.closed_trades]

    final_equity = engine.broker.equity_curve[-1]["equity"]
    total_return = calculate_total_return(equity_series, initial_capital=initial_cash)
    annualized = calculate_annualized_return(equity_series, annual_days=242)
    sharpe = calculate_sharpe_ratio(equity_series)
    max_dd = calculate_max_drawdown(equity_series)
    calmar = calculate_calmar_ratio(annualized, max_dd, risk_free_rate=0.015)
    win_rate, win_count, total_trades = calculate_win_rate(trade_returns)
    pl_ratio = calculate_profit_loss_ratio(trade_returns)

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "win_count": win_count,
        "total_trades": total_trades,
        "profit_loss_ratio": pl_ratio,
        "elapsed_sec": elapsed,
        "equity_curve": engine.broker.equity_curve,
    }


def main():
    config = load_config()
    symbols = list(config.get("etf_pool", {}).keys())
    initial_cash = config.get("initial_cash", 100000.0)
    data_source = config.get("data_source", "akshare")

    time_periods = [
        ("2024 全年", "20240101", "20241231"),
        ("2025 全年", "20250101", "20251231"),
        ("2024-2025 两年", "20240101", "20251231"),
    ]

    experiments = []

    # ── 1. BenchmarkStrategy ──
    for label, sd, ed in time_periods:
        experiments.append({
            "name": f"Benchmark (Buy & Hold) [{label}]",
            "cls": BenchmarkStrategy,
            "sd": sd, "ed": ed,
            "params": {},
        })

    # ── 2. RotationStrategy 参数矩阵 ──
    rotation_configs = [
        {"lookback_period": 10, "hold_period": 10, "top_n": 2, "use_efficiency": True},
        {"lookback_period": 10, "hold_period": 10, "top_n": 3, "use_efficiency": True},
        {"lookback_period": 20, "hold_period": 20, "top_n": 2, "use_efficiency": True},
        {"lookback_period": 20, "hold_period": 20, "top_n": 3, "use_efficiency": True},
        {"lookback_period": 20, "hold_period": 20, "top_n": 4, "use_efficiency": True},
        {"lookback_period": 30, "hold_period": 20, "top_n": 3, "use_efficiency": True},
        {"lookback_period": 20, "hold_period": 10, "top_n": 3, "use_efficiency": True},
        {"lookback_period": 20, "hold_period": 20, "top_n": 3, "use_efficiency": False},
    ]
    for rc in rotation_configs:
        mode = "效率轮动" if rc["use_efficiency"] else "纯动量轮动"
        tag = f"lb{rc['lookback_period']}_hp{rc['hold_period']}_n{rc['top_n']}"
        for label, sd, ed in time_periods:
            experiments.append({
                "name": f"Rotation({mode}) {tag} [{label}]",
                "cls": RotationStrategy,
                "sd": sd, "ed": ed,
                "params": rc,
            })

    # ── 3. DoubleMAStrategy ──
    ma_configs = [
        {"short_window": 5, "long_window": 20},
        {"short_window": 10, "long_window": 30},
        {"short_window": 5, "long_window": 60},
    ]
    for mc in ma_configs:
        tag = f"ma{mc['short_window']}_{mc['long_window']}"
        for label, sd, ed in time_periods:
            experiments.append({
                "name": f"DoubleMA {tag} [{label}]",
                "cls": DoubleMAStrategy,
                "sd": sd, "ed": ed,
                "params": mc,
            })

    # ── 4. TripleMomentumStrategy ──
    tm_configs = [
        {"bias_n": 24, "momentum_day": 25, "slope_n": 20, "hold_period": 20},
        {"bias_n": 20, "momentum_day": 20, "slope_n": 15, "hold_period": 15},
        {"bias_n": 30, "momentum_day": 30, "slope_n": 25, "hold_period": 20},
    ]
    for tc in tm_configs:
        tag = f"b{tc['bias_n']}_m{tc['momentum_day']}_s{tc['slope_n']}_h{tc['hold_period']}"
        for label, sd, ed in time_periods:
            experiments.append({
                "name": f"TripleMomentum {tag} [{label}]",
                "cls": TripleMomentumStrategy,
                "sd": sd, "ed": ed,
                "params": tc,
            })

    # ── Run all experiments ──
    results = []
    total = len(experiments)
    print(f"\n{'='*80}")
    print(f"  BreadFree 全策略回测分析")
    print(f"  共 {total} 组实验 | ETF池: {symbols}")
    print(f"  初始资金: ¥{initial_cash:,.0f}")
    print(f"{'='*80}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {exp['name']}")
        print("-" * 60)
        try:
            res = run_single_backtest(
                strategy_cls=exp["cls"],
                symbols=symbols,
                start_date=exp["sd"],
                end_date=exp["ed"],
                initial_cash=initial_cash,
                data_source=data_source,
                **exp["params"],
            )
            if res:
                res["name"] = exp["name"]
                results.append(res)
                print(f"  => 最终净值: ¥{res['final_equity']:,.2f} | 总收益: {res['total_return']:.2%} "
                      f"| 年化: {res['annualized_return']:.2%} | Sharpe: {res['sharpe_ratio']:.2f} "
                      f"| 最大回撤: {res['max_drawdown']:.2%} | 耗时: {res['elapsed_sec']:.1f}s")
            else:
                print("  => 无数据/回测失败")
        except Exception as e:
            print(f"  => 错误: {e}")

    # ── Generate report ──
    print(f"\n\n{'='*80}")
    print("                      综合收益报告")
    print(f"{'='*80}\n")

    if not results:
        print("无结果。")
        return

    df = pd.DataFrame([{
        "策略": r["name"],
        "最终净值": r["final_equity"],
        "总收益率": r["total_return"],
        "年化收益率": r["annualized_return"],
        "Sharpe": r["sharpe_ratio"],
        "Calmar": r["calmar_ratio"],
        "最大回撤": r["max_drawdown"],
        "胜率": r["win_rate"],
        "盈亏比": r["profit_loss_ratio"],
        "交易次数": r["total_trades"],
    } for r in results])

    # Format for display
    fmt = df.copy()
    fmt["最终净值"] = fmt["最终净值"].map(lambda x: f"¥{x:,.2f}")
    fmt["总收益率"] = fmt["总收益率"].map(lambda x: f"{x:.2%}")
    fmt["年化收益率"] = fmt["年化收益率"].map(lambda x: f"{x:.2%}")
    fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
    fmt["Calmar"] = fmt["Calmar"].map(lambda x: f"{x:.2f}")
    fmt["最大回撤"] = fmt["最大回撤"].map(lambda x: f"{x:.2%}")
    fmt["胜率"] = fmt["胜率"].map(lambda x: f"{x:.2%}")
    fmt["盈亏比"] = fmt["盈亏比"].map(lambda x: f"{x:.2f}")
    fmt["交易次数"] = fmt["交易次数"].astype(int)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    # ── 按时间段分别展示 ──
    for period_label, _, _ in time_periods:
        sub = fmt[fmt["策略"].str.contains(period_label)]
        if sub.empty:
            continue
        print(f"\n--- {period_label} ---")
        print(sub.to_string(index=False))

    # ── Best strategies per period ──
    print(f"\n\n{'='*80}")
    print("                    各时间段最佳策略排行")
    print(f"{'='*80}")

    for period_label, _, _ in time_periods:
        sub = df[df["策略"].str.contains(period_label)]
        if sub.empty:
            continue
        print(f"\n--- {period_label} (按年化收益率排序 Top 5) ---")
        top5 = sub.nlargest(5, "年化收益率")
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"  #{rank} {row['策略']}")
            print(f"      年化: {row['年化收益率']:.2%} | Sharpe: {row['Sharpe']:.2f} | "
                  f"回撤: {row['最大回撤']:.2%} | Calmar: {row['Calmar']:.2f}")

        print(f"\n--- {period_label} (按Sharpe排序 Top 5) ---")
        top5_sharpe = sub.nlargest(5, "Sharpe")
        for rank, (_, row) in enumerate(top5_sharpe.iterrows(), 1):
            print(f"  #{rank} {row['策略']}")
            print(f"      Sharpe: {row['Sharpe']:.2f} | 年化: {row['年化收益率']:.2%} | "
                  f"回撤: {row['最大回撤']:.2%}")

    # ── Save CSV ──
    csv_path = "output/backtest_all_results.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至: {csv_path}")


if __name__ == "__main__":
    main()
