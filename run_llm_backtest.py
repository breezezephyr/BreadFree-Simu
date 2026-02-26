#!/usr/bin/env python3
"""
BreadFree LLM策略回测：AgentStrategy + EffiAgentRotationStrategy
使用最近一个月实盘数据，同时运行Benchmark和RotationStrategy作为对照
"""
import sys, os, time, json
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


def main():
    config = load_config()
    symbols = list(config.get("etf_pool", {}).keys())
    initial_cash = config.get("initial_cash", 100000.0)
    data_source = config.get("data_source", "akshare")

    start_date = "20260127"
    end_date = "20260226"

    print(f"\n{'#'*70}")
    print(f"  BreadFree LLM策略回测 — 最近一个月实盘数据")
    print(f"  区间: {start_date} ~ {end_date}")
    print(f"  ETF池: {symbols}")
    print(f"  初始资金: ¥{initial_cash:,.0f}")
    print(f"  LLM Provider: {os.environ.get('LLM_PROVIDER', 'volcano')}")
    print(f"{'#'*70}")

    results = []

    # ── 1. Benchmark ──
    r = run_backtest(BenchmarkStrategy, symbols, start_date, end_date,
                     initial_cash, data_source,
                     label="① Benchmark (Buy & Hold)")
    if r: results.append(r)

    # ── 2. RotationStrategy (效率轮动 最优参数) ──
    r = run_backtest(RotationStrategy, symbols, start_date, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20, top_n=2, use_efficiency=True,
                     label="② RotationStrategy (效率轮动 lb20_hp20_n2)")
    if r: results.append(r)

    # ── 3. RotationStrategy (纯动量) ──
    r = run_backtest(RotationStrategy, symbols, start_date, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20, top_n=3, use_efficiency=False,
                     label="③ RotationStrategy (纯动量 lb20_hp20_n3)")
    if r: results.append(r)

    # ── 4. AgentStrategy (LLM多Agent投委会) ──
    r = run_backtest(AgentStrategy, symbols, start_date, end_date,
                     initial_cash, data_source,
                     label="④ AgentStrategy (LLM多Agent投委会)")
    if r: results.append(r)

    # ── 5. EffiAgentRotationStrategy (LLM效率轮动) ──
    r = run_backtest(EffiAgentRotationStrategy, symbols, start_date, end_date,
                     initial_cash, data_source,
                     lookback_period=20, hold_period=20,
                     label="⑤ EffiAgentRotationStrategy (LLM效率轮动)")
    if r: results.append(r)

    # ── 汇总对比 ──
    if results:
        print(f"\n\n{'='*70}")
        print(f"               最近一个月策略对比汇总")
        print(f"               {start_date} ~ {end_date}")
        print(f"{'='*70}\n")

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "label": "策略",
            "final_equity": "最终净值",
            "total_return": "总收益率",
            "annualized_return": "年化收益率",
            "sharpe": "Sharpe",
            "calmar": "Calmar",
            "max_drawdown": "最大回撤",
            "win_rate": "胜率",
            "pl_ratio": "盈亏比",
            "total_trades": "交易次数",
            "elapsed": "耗时(s)",
        })

        display_cols = ["策略", "最终净值", "总收益率", "年化收益率", "Sharpe",
                        "Calmar", "最大回撤", "胜率", "盈亏比", "交易次数", "耗时(s)"]
        fmt = df[display_cols].copy()
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

        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 220)
        print(fmt.to_string(index=False))

        # Save
        os.makedirs("output", exist_ok=True)
        df[display_cols].to_csv("output/llm_backtest_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n结果已保存至: output/llm_backtest_results.csv")


if __name__ == "__main__":
    main()
