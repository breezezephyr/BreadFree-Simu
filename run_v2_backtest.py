#!/usr/bin/env python3
"""V2 AgentStrategy 回测对比"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd, yaml
from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.agent_strategy import AgentStrategy
from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy
from breadfree.utils.metrics import (
    calculate_total_return, calculate_max_drawdown,
    calculate_sharpe_ratio, calculate_annualized_return,
    calculate_profit_loss_ratio, calculate_win_rate, calculate_calmar_ratio,
)

def load_config():
    with open(os.path.join(os.path.dirname(__file__), "breadfree", "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def run(cls, symbols, sd, ed, cash, ds, label, **kw):
    print(f"\n{'='*60}\n  {label}\n  {sd} ~ {ed}\n{'='*60}")
    engine = BacktestEngine(strategy_cls=cls, symbols=symbols, start_date=sd, end_date=ed,
                            initial_cash=cash, asset_type="stock", lot_size=100, data_source=ds, **kw)
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0
    if not engine.broker.equity_curve:
        print("  => FAILED"); return None
    es = pd.Series([d["equity"] for d in engine.broker.equity_curve])
    tr = [t["return_pct"] for t in engine.broker.closed_trades]
    fe = engine.broker.equity_curve[-1]["equity"]
    ret = calculate_total_return(es, initial_capital=cash)
    ann = calculate_annualized_return(es, annual_days=242)
    sh = calculate_sharpe_ratio(es)
    mdd = calculate_max_drawdown(es)
    cal = calculate_calmar_ratio(ann, mdd, risk_free_rate=0.015)
    wr, wc, tt = calculate_win_rate(tr)
    plr = calculate_profit_loss_ratio(tr)
    print(f"  净值 ¥{fe:,.2f} | 收益 {ret:.2%} | 年化 {ann:.2%} | Sharpe {sh:.2f} | 回撤 {mdd:.2%} | 耗时 {elapsed:.1f}s")
    return {"label": label, "final_equity": fe, "total_return": ret, "annualized": ann,
            "sharpe": sh, "calmar": cal, "max_dd": mdd, "win_rate": wr, "trades": tt, "plr": plr, "time": elapsed}

def main():
    cfg = load_config()
    syms = list(cfg.get("etf_pool", {}).keys())
    cash = cfg.get("initial_cash", 100000.0)
    ds = cfg.get("data_source", "akshare")

    periods = [
        ("3个月实盘", "20251127", "20260226"),
        ("2025全年", "20250101", "20251231"),
    ]
    results = []

    for plabel, sd, ed in periods:
        print(f"\n\n{'#'*60}\n  === {plabel} ===\n{'#'*60}")

        r = run(BenchmarkStrategy, syms, sd, ed, cash, ds, f"Benchmark [{plabel}]")
        if r: results.append(r)

        r = run(RotationStrategy, syms, sd, ed, cash, ds, f"Rotation效率轮动 lb20n2 [{plabel}]",
                lookback_period=20, hold_period=20, top_n=2, use_efficiency=True)
        if r: results.append(r)

        r = run(RotationStrategy, syms, sd, ed, cash, ds, f"Rotation效率轮动 lb20n3 [{plabel}]",
                lookback_period=20, hold_period=20, top_n=3, use_efficiency=True)
        if r: results.append(r)

        r = run(AgentStrategyV2, syms, sd, ed, cash, ds, f"AgentV2(LLM投委会v2) [{plabel}]",
                lookback_period=20, hold_period=20, top_n=3)
        if r: results.append(r)

        r = run(EffiAgentRotationStrategy, syms, sd, ed, cash, ds, f"EffiAgent(LLM效率轮动) [{plabel}]",
                lookback_period=20, hold_period=20)
        if r: results.append(r)

    if results:
        print(f"\n\n{'='*60}\n  综合对比\n{'='*60}\n")
        df = pd.DataFrame(results)
        fmt = df.copy()
        fmt["final_equity"] = fmt["final_equity"].map(lambda x: f"¥{x:,.2f}")
        fmt["total_return"] = fmt["total_return"].map(lambda x: f"{x:.2%}")
        fmt["annualized"] = fmt["annualized"].map(lambda x: f"{x:.2%}")
        fmt["sharpe"] = fmt["sharpe"].map(lambda x: f"{x:.2f}")
        fmt["calmar"] = fmt["calmar"].map(lambda x: f"{x:.2f}")
        fmt["max_dd"] = fmt["max_dd"].map(lambda x: f"{x:.2%}")
        fmt["win_rate"] = fmt["win_rate"].map(lambda x: f"{x:.2%}")
        fmt["plr"] = fmt["plr"].map(lambda x: f"{x:.2f}")
        fmt["time"] = fmt["time"].map(lambda x: f"{x:.1f}s")
        fmt.columns = ["策略", "净值", "总收益", "年化", "Sharpe", "Calmar", "最大回撤", "胜率", "交易", "盈亏比", "耗时"]
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 50)

        for plabel, _, _ in periods:
            sub = fmt[fmt["策略"].str.contains(plabel)]
            if not sub.empty:
                print(f"\n--- {plabel} ---")
                print(sub.to_string(index=False))

if __name__ == "__main__":
    main()
