"""
BreadFree 全策略批量回测对比 — 1年/半年/3个月三时间窗口

用法:
    uv run python batch_compare.py              # 含 LLM 策略（慢）
    uv run python batch_compare.py --quant-only # 仅纯量化，快速出结果

耗时说明：AgentV2/EffiA 每个调仓日会打 2～3 次 LLM API（约 30～50s/次），
3 个月 × hold_period=20 ≈ 3 次调仓 × 2 策略 ≈ 6～15 分钟仅 LLM 部分。欲快速对比请用 --quant-only。
"""

import argparse
import sys, os, time
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.data.database import get_db_manager
from breadfree.utils.metrics import (
    calculate_total_return, calculate_max_drawdown,
    calculate_sharpe_ratio, calculate_annualized_return,
    calculate_calmar_ratio, calculate_win_rate, calculate_profit_loss_ratio,
)

from breadfree.strategies.ma_strategy import DoubleMAStrategy
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.triple_momentum_strategy import TripleMomentumStrategy
from breadfree.strategies.dynamic_rotation_strategy import DynamicRotationStrategy

from breadfree.utils.llm_client import get_llm_token_sum, reset_llm_token_sum

HAS_LLM = bool(os.getenv("ARK_API_KEY") or os.getenv("NVIDIA_API_KEY"))
if HAS_LLM:
    from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
    from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy


# ═══════════════════════════════════════════════════════════════
# 策略清单
# ═══════════════════════════════════════════════════════════════

def build_strategy_list(quant_only: bool = False):
    strategies = [
        ("Benchmark (买入持有)",       BenchmarkStrategy, {}),
        ("DoubleMA (双均线交叉)",      DoubleMAStrategy,  {}),
        ("Rotation lb20_n3 (效率轮动Top3)", RotationStrategy,
         {"lookback_period": 20, "hold_period": 20, "top_n": 3}),
        ("Rotation lb20_n2 (效率轮动Top2)", RotationStrategy,
         {"lookback_period": 20, "hold_period": 20, "top_n": 2}),
        ("Rotation lb20_n5 (效率轮动Top5)", RotationStrategy,
         {"lookback_period": 20, "hold_period": 20, "top_n": 5}),
        ("TripleMomentum (三因子动量)", TripleMomentumStrategy,
         {"lookback_period": 20, "hold_period": 20, "top_n": 3}),
        ("DynamicRotation (主动+动态)", DynamicRotationStrategy,
         {"lookback_period": 20, "top_n": 3, "enable_discovery": False}),
    ]
    if HAS_LLM and not quant_only:
        strategies.extend([
            ("AgentV2 (Bull-Bear辩证)", AgentStrategyV2,
             {"lookback_period": 20, "hold_period": 20, "top_n": 3}),
            ("EffiA (LLM轻量轮动)",     EffiAgentRotationStrategy,
             {"lookback_period": 20, "hold_period": 20}),
        ])
    return strategies


# ═══════════════════════════════════════════════════════════════
# 时间窗口
# ═══════════════════════════════════════════════════════════════

WINDOWS = [
    ("3个月","20251001", "20251231"),
]


# ═══════════════════════════════════════════════════════════════
# 单次回测
# ═══════════════════════════════════════════════════════════════

def run_one(cls, params, symbols, start, end, cash=100000.0):
    get_db_manager().clear_cache()
    try:
        engine = BacktestEngine(
            strategy_cls=cls, symbols=symbols,
            start_date=start, end_date=end,
            initial_cash=cash, **params,
        )
        engine.run()
    except Exception as e:
        return None, str(e)

    if not engine.broker.equity_curve:
        return None, "无权益数据"

    eq = pd.Series([d["equity"] for d in engine.broker.equity_curve])
    tr = [t["return_pct"] for t in engine.broker.closed_trades]
    total_ret  = calculate_total_return(eq, initial_capital=cash)
    annual_ret = calculate_annualized_return(eq, annual_days=242)
    max_dd     = calculate_max_drawdown(eq)
    sharpe     = calculate_sharpe_ratio(eq)
    calmar     = calculate_calmar_ratio(annual_ret, max_dd, risk_free_rate=0.015)
    wr, wins, tot = calculate_win_rate(tr)
    plr        = calculate_profit_loss_ratio(tr)

    return {
        "final": engine.broker.equity_curve[-1]["equity"],
        "ret": total_ret, "ann": annual_ret,
        "mdd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "wr": wr, "wins": wins, "trades": tot, "plr": plr,
    }, None


# ═══════════════════════════════════════════════════════════════
# 打印对比表格
# ═══════════════════════════════════════════════════════════════

def fmt_pct(v):
    return f"{v:+.2%}" if v else "N/A"

def print_table(window_name, rows):
    print(f"\n{'━'*115}")
    print(f"  📊 时间窗口: {window_name}")
    print(f"{'━'*115}")
    hdr = (f"  {'#':<3s}{'策略':<36s}{'终值':>10s}{'总收益':>9s}{'年化':>9s}"
           f"{'Sharpe':>8s}{'MaxDD':>9s}{'Calmar':>8s}"
           f"{'胜率':>7s}{'盈亏比':>7s}{'交易':>5s}")
    print(hdr)
    print(f"  {'─'*112}")

    valid = sorted([r for r in rows if r["m"]], key=lambda r: r["m"]["ret"], reverse=True)
    for i, r in enumerate(valid, 1):
        m = r["m"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
        line = (
            f"  {medal}{i:<2d}"
            f"{r['name']:<36s}"
            f"¥{m['final']:>9,.0f}"
            f"{m['ret']:>+8.2%}"
            f"{m['ann']:>+8.2%}"
            f"{m['sharpe']:>8.2f}"
            f"{m['mdd']:>+8.2%}"
            f"{m['calmar']:>8.2f}"
            f"{m['wr']:>6.1%}"
            f"{m['plr']:>7.2f}"
            f"{m['trades']:>5d}"
        )
        print(line)

    errs = [r for r in rows if r["err"]]
    if errs:
        print(f"  {'─'*112}")
        for r in errs:
            print(f"    ✗ {r['name']}: {r['err']}")

    if valid:
        best_ret = valid[0]
        best_sh  = max(valid, key=lambda r: r["m"]["sharpe"])
        best_dd  = max(valid, key=lambda r: r["m"]["mdd"])
        print(f"  {'─'*112}")
        print(f"  最高收益: {best_ret['name']} ({best_ret['m']['ret']:+.2%})")
        print(f"  最高Sharpe: {best_sh['name']} ({best_sh['m']['sharpe']:.2f})")
        print(f"  最小回撤: {best_dd['name']} ({best_dd['m']['mdd']:+.2%})")
    print(f"{'━'*115}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BreadFree 全策略批量回测")
    parser.add_argument("--quant-only", action="store_true", help="仅跑纯量化策略，跳过 AgentV2/EffiA，速度快")
    args = parser.parse_args()

    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "breadfree", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    symbols = list(cfg.get("etf_pool", {}).keys())

    strategies = build_strategy_list(quant_only=args.quant_only)

    print(f"╔{'═'*60}╗")
    print(f"║  BreadFree 全策略批量回测对比                              ║")
    print(f"║  标的池: {len(symbols)} 个 (ETF+个股)                               ║")
    print(f"║  策略数: {len(strategies)} 个                                          ║")
    print(f"║  时间窗口: 1年 / 半年 / 3个月                              ║")
    if args.quant_only:
        print(f"║  --quant-only: 仅纯量化，已跳过 LLM 策略                           ║")
    elif HAS_LLM:
        print(f"║  ✓ LLM API 可用, 含 AgentV2 + EffiA                       ║")
    else:
        print(f"║  ✗ 无 LLM Key, 跳过 AgentV2 / EffiA                       ║")
    print(f"╚{'═'*60}╝\n")

    all_results = {}
    if HAS_LLM:
        reset_llm_token_sum()

    for w_name, w_start, w_end in WINDOWS:
        print(f"\n{'='*70}")
        print(f"  🔄 开始 [{w_name}] 窗口回测  ({w_start} ~ {w_end})")
        print(f"{'='*70}")

        rows = []
        for idx, (s_name, s_cls, s_params) in enumerate(strategies, 1):
            label = f"[{w_name}] [{idx}/{len(strategies)}] {s_name}"
            print(f"  ▸ {label} ...", end=" ", flush=True)
            t0 = time.time()
            m, err = run_one(s_cls, s_params, symbols, w_start, w_end)
            elapsed = time.time() - t0
            if err:
                print(f"✗ {err} ({elapsed:.1f}s)")
            else:
                print(f"✓ ret={m['ret']:+.2%} sharpe={m['sharpe']:.2f} "
                      f"mdd={m['mdd']:+.2%} ({elapsed:.1f}s)")
            rows.append({"name": s_name, "m": m, "err": err})

        all_results[w_name] = rows
        print_table(w_name, rows)

    # 汇总输出
    print(f"\n\n{'▓'*115}")
    print(f"  📋 全策略 × 全窗口 汇总矩阵")
    print(f"{'▓'*115}")

    s_names = [s[0] for s in strategies]
    print(f"\n  {'策略':<36s}", end="")
    for w_name, _, _ in WINDOWS:
        print(f"{'收益('+w_name+')':>12s}{'Sharpe':>8s}{'MaxDD':>9s}", end="")
    print()
    print(f"  {'─'*36}", end="")
    for _ in WINDOWS:
        print(f"{'─'*29}", end="")
    print()

    for s_name in s_names:
        print(f"  {s_name:<36s}", end="")
        for w_name, _, _ in WINDOWS:
            rows = all_results.get(w_name, [])
            r = next((r for r in rows if r["name"] == s_name), None)
            if r and r["m"]:
                m = r["m"]
                print(f"{m['ret']:>+11.2%}{m['sharpe']:>8.2f}{m['mdd']:>+9.2%}", end="")
            else:
                print(f"{'N/A':>11s}{'N/A':>8s}{'N/A':>9s}", end="")
        print()

    print(f"\n{'▓'*115}")
    if HAS_LLM:
        s = get_llm_token_sum()
        print(f"  📊 本轮 LLM 跑测 token 消耗: total_tokens={s['total_tokens']}, call_count={s['call_count']}")
    print("  完成!\n")


if __name__ == "__main__":
    main()
