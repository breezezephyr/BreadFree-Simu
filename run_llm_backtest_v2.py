#!/usr/bin/env python3
"""
BreadFree LLM策略回测 v2
AgentStrategy + EffiAgentRotationStrategy 完整对比
使用最近3个月实盘数据（2025-11-27 ~ 2026-02-26），确保充足的调仓周期
"""
import sys, os, time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yaml

from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.agent_strategy import AgentStrategy
from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy
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


def run_backtest(strategy_cls, symbols, start_date, end_date, initial_cash,
                 data_source, lot_size=100, label="", **kwargs):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  区间: {start_date} ~ {end_date} | 初始资金: ¥{initial_cash:,.0f}")
    print(f"{'='*70}")

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
        print("  => 无数据/回测失败")
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

    print(f"\n  ── 回测结果 ──")
    print(f"  最终净值:   ¥{final_equity:,.2f}")
    print(f"  总收益率:   {total_return:.2%}")
    print(f"  年化收益率: {annualized:.2%}")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  Calmar:     {calmar:.2f}")
    print(f"  最大回撤:   {max_dd:.2%}")
    print(f"  胜率:       {win_rate:.2%} ({win_count}/{total_trades})")
    print(f"  盈亏比:     {pl_ratio:.2f}")
    print(f"  耗时:       {elapsed:.1f}s")

    return {
        "label": label,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "win_count": win_count,
        "total_trades": total_trades,
        "pl_ratio": pl_ratio,
        "elapsed": elapsed,
    }


def print_summary(results, title, period):
    if not results:
        return
    print(f"\n\n{'='*70}")
    print(f"  {title}")
    print(f"  {period}")
    print(f"{'='*70}\n")

    df = pd.DataFrame(results)
    display_cols = ["label", "final_equity", "total_return", "annualized_return",
                    "sharpe", "calmar", "max_drawdown", "win_rate", "pl_ratio",
                    "total_trades", "elapsed"]
    fmt = df[display_cols].copy()
    fmt.columns = ["策略", "最终净值", "总收益率", "年化收益率", "Sharpe",
                   "Calmar", "最大回撤", "胜率", "盈亏比", "交易次数", "耗时(s)"]
    fmt["最终净值"] = fmt["最终净值"].map(lambda x: f"¥{x:,.2f}")
    fmt["总收益率"] = fmt["总收益率"].map(lambda x: f"{x:.2%}")
    fmt["年化收益率"] = fmt["年化收益率"].map(lambda x: f"{x:.2%}")
    fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
    fmt["Calmar"] = fmt["Calmar"].map(lambda x: f"{x:.2f}")
    fmt["最大回撤"] = fmt["最大回撤"].map(lambda x: f"{x:.2%}")
    fmt["胜率"] = fmt["胜率"].map(lambda x: f"{x:.2%}")
    fmt["盈亏比"] = fmt["盈亏比"].map(lambda x: f"{x:.2f}")
    fmt["交易次数"] = fmt["交易次数"].astype(int)
    fmt["耗时(s)"] = fmt["耗时(s)"].map(lambda x: f"{x:.1f}")

    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 220)
    print(fmt.to_string(index=False))
    return df


def main():
    config = load_config()
    symbols = list(config.get("etf_pool", {}).keys())
    initial_cash = config.get("initial_cash", 100000.0)
    data_source = config.get("data_source", "akshare")

    # 最近3个月：足够的调仓周期
    start_3m = "20251127"
    end_date = "20260226"

    print(f"\n{'#'*70}")
    print(f"  BreadFree LLM策略 完整回测对比")
    print(f"  ETF池: {symbols}")
    print(f"  初始资金: ¥{initial_cash:,.0f}")
    print(f"  LLM Provider: volcano (Ark)")
    print(f"{'#'*70}")

    results = []

    # ── 对照组：纯量化策略 ──
    print(f"\n\n{'*'*70}")
    print(f"  第一部分：对照组（纯量化策略）")
    print(f"  区间: {start_3m} ~ {end_date} (约3个月)")
    print(f"{'*'*70}")

    r = run_backtest(BenchmarkStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     label="Benchmark (Buy & Hold)")
    if r: results.append(r)

    r = run_backtest(RotationStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20, top_n=2, use_efficiency=True,
                     label="Rotation 效率轮动 (lb20 hp20 n2)")
    if r: results.append(r)

    r = run_backtest(RotationStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20, top_n=3, use_efficiency=True,
                     label="Rotation 效率轮动 (lb20 hp20 n3)")
    if r: results.append(r)

    r = run_backtest(RotationStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     lookback_period=10, hold_period=10, top_n=3, use_efficiency=True,
                     label="Rotation 效率轮动 (lb10 hp10 n3)")
    if r: results.append(r)

    r = run_backtest(TripleMomentumStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     bias_n=24, momentum_day=25, slope_n=20, hold_period=20,
                     label="TripleMomentum (b24 m25 s20 h20)")
    if r: results.append(r)

    # ── LLM策略 ──
    print(f"\n\n{'*'*70}")
    print(f"  第二部分：LLM智能体策略")
    print(f"  区间: {start_3m} ~ {end_date} (约3个月)")
    print(f"{'*'*70}")

    # AgentStrategy: 每日调用3次LLM，只分析第一只ETF(510300)
    r = run_backtest(AgentStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     label="AgentStrategy (LLM投委会)")
    if r: results.append(r)

    # EffiAgentRotationStrategy: 仅在调仓日调用LLM
    r = run_backtest(EffiAgentRotationStrategy, symbols, start_3m, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20,
                     label="EffiAgent (LLM效率轮动)")
    if r: results.append(r)

    # ── 汇总 ──
    summary_df = print_summary(results,
                               "全策略对比汇总（含LLM策略）",
                               f"区间: {start_3m} ~ {end_date}")

    # Save
    if summary_df is not None:
        os.makedirs("output", exist_ok=True)
        pd.DataFrame(results).to_csv("output/llm_backtest_v2_results.csv",
                                     index=False, encoding="utf-8-sig")
        print(f"\n结果已保存至: output/llm_backtest_v2_results.csv")


if __name__ == "__main__":
    main()
