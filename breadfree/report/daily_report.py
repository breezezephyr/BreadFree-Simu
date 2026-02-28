"""
每日多策略决策报告生成器

功能:
    1. 计算全池 25 标的最新多因子效率分, 输出 Top-N 决策表
    2. 运行 3 种策略回测 (RotationStrategy / AgentStrategyV2 / EffiA)
    3. 每种策略: 排名表 + 文本总结 + 收益曲线
    4. 生成对比收益曲线 (3 条线叠加)
    5. 组装 HTML 邮件, 通过 email_reporter 发送
"""

import io
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from breadfree.utils.config import get_config
from breadfree.utils.logger import get_logger
from breadfree.utils.email_reporter import send_report_email
from breadfree.utils.metrics import (
    calculate_efficiency_metrics, calculate_total_return,
    calculate_max_drawdown, calculate_sharpe_ratio, calculate_annualized_return,
)
from breadfree.data.data_fetcher import DataFetcher
from breadfree.data.database import get_db_manager
from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy

logger = get_logger(__name__)

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")

STRATEGY_META = {
    "RotationStrategy": {
        "cls": RotationStrategy,
        "label": "RotationStrategy (纯量化效率轮动)",
        "color": "#c23531",
        "desc": "纯量化多因子策略。基于效率分 = (动量/波动率)×R² 进行标的排名，"
                "结合动量加速度和回撤惩罚进行多因子合成，等权配置 Top-N 标的，"
                "每 hold_period 天调仓一次。无需 LLM，执行速度最快。",
        "needs_llm": False,
    },
    "AgentStrategyV2": {
        "cls": AgentStrategyV2,
        "label": "AgentStrategyV2 (LLM 辩证决策)",
        "color": "#2f4554",
        "desc": "量化锚定 + LLM 精调。QuantEngine 计算效率分选出 Top-N 候选，"
                "Analyst LLM 根据多周期动量一致性和市场情报分配权重，"
                "RiskMgr LLM 风控审核微调。LLM 只能在量化候选池内调仓，"
                "不可引入新标的，总投资度 85%-95%。",
        "needs_llm": True,
    },
    "EffiA": {
        "cls": EffiAgentRotationStrategy,
        "label": "EffiA (LLM 轻量轮动)",
        "color": "#61a0a8",
        "desc": "轻量版 LLM 轮动。DataPrep 计算效率分筛选 Top-3 候选，"
                "Analyst LLM 在候选池内分配权重（至少 2 只，单只≤60%），"
                "RiskMgr LLM 微调。架构比 V2 更简洁，响应更快，"
                "适合低延迟场景。",
        "needs_llm": True,
    },
}


# ═══════════════════════════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════════════════════════

def _get_etf_pool() -> dict:
    cfg = get_config()
    return dict(cfg.get("etf_pool", {"510300": "沪深300ETF"}))


def _fetch_latest_prices(symbols: list, lookback_days: int = 60) -> dict:
    now = datetime.now(TZ_SHANGHAI)
    end_date = now.strftime("%Y%m%d")
    start_date = (now - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    fetcher = DataFetcher(data_dir="breadfree/data/cache", data_source="akshare")
    db = get_db_manager()
    db.preload_symbols(symbols, start_date, end_date)

    price_map = {}
    for symbol in symbols:
        df = db.get_preloaded(symbol)
        if df is None or df.empty:
            df = db.get_daily_data(symbol, start_date, end_date)
        if df.empty:
            df = fetcher.fetch_a_stock_daily(symbol, start_date, end_date)
        if df.empty:
            continue

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df.set_index("trade_date", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        closes = df["close"].dropna().tolist()
        if closes:
            price_map[symbol] = closes

    return price_map


def calc_top_n_scores(top_n: int = 5, lookback: int = 20) -> list:
    pool = _get_etf_pool()
    symbols = list(pool.keys())
    price_map = _fetch_latest_prices(symbols, lookback_days=lookback * 3)

    scored = []
    for symbol in symbols:
        prices = price_map.get(symbol)
        if not prices or len(prices) < lookback:
            continue
        metrics = calculate_efficiency_metrics(prices, lookback)
        if metrics is None:
            continue
        metrics["symbol"] = symbol
        metrics["name"] = pool.get(symbol, symbol)
        scored.append(metrics)

    scored.sort(key=lambda x: x["efficiency"], reverse=True)
    for i, item in enumerate(scored[:top_n], 1):
        item["rank"] = i

    return scored[:top_n]


# ═══════════════════════════════════════════════════════════════
# 回测运行 & 性能摘要
# ═══════════════════════════════════════════════════════════════

def _run_backtest(strategy_name: str, backtest_days: int,
                  top_n: int = 5, lookback: int = 20) -> Optional[dict]:
    """运行单个策略回测, 返回 {equity_curve, metrics, strategy_name}"""
    meta = STRATEGY_META.get(strategy_name)
    if not meta:
        return None

    has_llm_keys = bool(os.getenv("ARK_API_KEY") or os.getenv("NVIDIA_API_KEY"))
    if meta["needs_llm"] and not has_llm_keys:
        logger.warning(f"[Report] {strategy_name} 需要 LLM API Key, 跳过")
        return None

    now = datetime.now(TZ_SHANGHAI)
    end_date = now.strftime("%Y%m%d")
    start_date = (now - timedelta(days=backtest_days)).strftime("%Y%m%d")

    cfg = get_config()
    symbols = list(cfg.get("etf_pool", {"510300": "沪深300ETF"}).keys())
    initial_cash = cfg.get("initial_cash", 100000.0)

    kwargs = {"lookback_period": lookback, "hold_period": 20, "top_n": top_n}

    get_db_manager().clear_cache()

    logger.info(f"[Report] 运行 {strategy_name} 回测 ({start_date}~{end_date})...")
    try:
        engine = BacktestEngine(
            strategy_cls=meta["cls"],
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            **kwargs,
        )
        engine.run()
    except Exception as e:
        logger.error(f"[Report] {strategy_name} 回测异常: {e}")
        return None

    if not engine.broker.equity_curve:
        logger.warning(f"[Report] {strategy_name} 无权益数据")
        return None

    eq = pd.Series([d["equity"] for d in engine.broker.equity_curve])
    dates = [d["date"] for d in engine.broker.equity_curve]
    equities = [d["equity"] for d in engine.broker.equity_curve]
    trade_returns = [t["return_pct"] for t in engine.broker.closed_trades]

    total_ret = calculate_total_return(eq, initial_capital=initial_cash)
    annual_ret = calculate_annualized_return(eq, annual_days=242)
    max_dd = calculate_max_drawdown(eq)
    sharpe = calculate_sharpe_ratio(eq)

    final_equity = equities[-1] if equities else initial_cash
    pool = _get_etf_pool()

    holdings = []
    for sym, pos in engine.broker.positions.items():
        qty = pos.quantity
        avg = pos.avg_price
        last_price = pos.avg_price
        if sym in engine.data_map and not engine.data_map[sym].empty:
            last_price = float(engine.data_map[sym]["close"].iloc[-1])
        mv = qty * last_price
        pnl = (last_price - avg) * qty
        pnl_pct = (last_price / avg - 1) if avg > 0 else 0
        holdings.append({
            "symbol": sym, "name": pool.get(sym, sym),
            "quantity": qty, "avg_price": avg,
            "last_price": last_price, "market_value": mv,
            "pnl": pnl, "pnl_pct": pnl_pct,
            "weight": mv / final_equity if final_equity > 0 else 0,
        })
    holdings.sort(key=lambda h: h["market_value"], reverse=True)

    recent_trades = []
    tx = engine.broker.transaction_history
    for t in tx[-8:]:
        d = t["date"]
        recent_trades.append({
            "date": d.strftime("%m-%d") if hasattr(d, "strftime") else str(d)[:10],
            "action": t["action"],
            "symbol": t["symbol"],
            "name": pool.get(t["symbol"], t["symbol"]),
            "price": t["price"],
            "quantity": t["quantity"],
        })

    closed = engine.broker.closed_trades
    top_wins = sorted([c for c in closed if c["return_pct"] > 0],
                      key=lambda c: c["pnl"], reverse=True)[:3]
    top_losses = sorted([c for c in closed if c["return_pct"] < 0],
                        key=lambda c: c["pnl"])[:3]

    return {
        "strategy_name": strategy_name,
        "label": meta["label"],
        "color": meta["color"],
        "desc": meta["desc"],
        "dates": dates,
        "equities": equities,
        "initial_cash": initial_cash,
        "metrics": {
            "total_return": total_ret,
            "annualized_return": annual_ret,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "total_trades": len(trade_returns),
        },
        "holdings": holdings,
        "recent_trades": recent_trades,
        "top_wins": top_wins,
        "top_losses": top_losses,
        "cash": engine.broker.cash,
        "cash_pct": engine.broker.cash / final_equity if final_equity > 0 else 1.0,
    }


# ═══════════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════════

def _generate_comparison_chart(results: List[dict]) -> bytes:
    """生成多策略对比收益曲线 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    initial_cash = results[0]["initial_cash"] if results else 100000

    for r in results:
        ax.plot(r["dates"], r["equities"], color=r["color"],
                linewidth=1.8, label=r["strategy_name"])

    ax.axhline(y=initial_cash, color="#999", linestyle="--", linewidth=0.8, label="Baseline")
    ax.set_title("Multi-Strategy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (CNY)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_single_chart(result: dict) -> bytes:
    """生成单策略收益曲线 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 4))
    initial = result["initial_cash"]
    ret = result["metrics"]["total_return"] * 100

    ax.plot(result["dates"], result["equities"], color=result["color"],
            linewidth=1.8, label="Equity")
    ax.fill_between(result["dates"], initial, result["equities"],
                    alpha=0.08, color=result["color"])
    ax.axhline(y=initial, color="#999", linestyle="--", linewidth=0.8)

    ax.set_title(f"{result['strategy_name']}  Return: {ret:+.2f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (CNY)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# HTML 报告
# ═══════════════════════════════════════════════════════════════

def _signal_html(eff: float) -> str:
    if eff >= 2.0:
        return '<span style="color:#c23531;font-weight:bold">&#9650; 强势</span>'
    elif eff >= 1.0:
        return '<span style="color:#d48806;font-weight:bold">&#9650; 看多</span>'
    elif eff >= 0:
        return '<span style="color:#666">&#9644; 中性</span>'
    else:
        return '<span style="color:#389e0d">&#9660; 看空</span>'


def _build_ranking_table(scores: list) -> str:
    rows = ""
    for item in scores:
        rows += f"""
        <tr>
            <td style="text-align:center">{item['rank']}</td>
            <td>{item['symbol']}</td>
            <td>{item['name']}</td>
            <td style="text-align:right">{item['close']:.3f}</td>
            <td style="text-align:right">{item['momentum'] * 100:+.2f}%</td>
            <td style="text-align:right">{item['volatility'] * 100:.2f}%</td>
            <td style="text-align:right">{item['r2']:.2f}</td>
            <td style="text-align:right;font-weight:bold">{item['efficiency']:.2f}</td>
            <td style="text-align:center">{_signal_html(item['efficiency'])}</td>
        </tr>"""
    return f"""
    <table style="border-collapse:collapse; width:100%; font-size:13px;" border="1" cellpadding="6">
    <thead style="background:#f5f5f5;">
    <tr><th>排名</th><th>代码</th><th>名称</th><th>最新价</th>
        <th>动量</th><th>波动率</th><th>R&sup2;</th><th>效率分</th><th>信号</th></tr>
    </thead><tbody>{rows}</tbody></table>"""


def _build_metrics_bar(m: dict) -> str:
    return (
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;font-size:13px;">'
        f'<span>收益 <b>{m["total_return"]:.2%}</b></span>'
        f'<span>年化 <b>{m["annualized_return"]:.2%}</b></span>'
        f'<span>Sharpe <b>{m["sharpe_ratio"]:.2f}</b></span>'
        f'<span>最大回撤 <b>{m["max_drawdown"]:.2%}</b></span>'
        f'<span>交易 <b>{m["total_trades"]}</b>次</span>'
        f'</div>')


def _build_holdings_table(result: dict) -> str:
    holdings = result.get("holdings", [])
    if not holdings:
        cash_pct = result.get("cash_pct", 1.0)
        return f'<p style="font-size:13px;color:#888;">全部现金 ({cash_pct:.0%})</p>'

    pool = _get_etf_pool()
    rows = ""
    for h in holdings:
        pnl_color = "#c23531" if h["pnl"] >= 0 else "#389e0d"
        rows += f"""
        <tr>
            <td>{h['symbol']}</td><td>{h['name']}</td>
            <td style="text-align:right">{h['quantity']}</td>
            <td style="text-align:right">{h['avg_price']:.3f}</td>
            <td style="text-align:right">{h['last_price']:.3f}</td>
            <td style="text-align:right">{h['market_value']:,.0f}</td>
            <td style="text-align:right;color:{pnl_color}">{h['pnl']:+,.0f} ({h['pnl_pct']:+.2%})</td>
            <td style="text-align:right">{h['weight']:.1%}</td>
        </tr>"""

    cash = result.get("cash", 0)
    cash_pct = result.get("cash_pct", 0)
    rows += f"""
    <tr style="background:#fafafa;">
        <td colspan="5" style="text-align:right"><i>现金</i></td>
        <td style="text-align:right">{cash:,.0f}</td>
        <td></td><td style="text-align:right">{cash_pct:.1%}</td>
    </tr>"""

    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:12px;" border="1" cellpadding="4">
    <thead style="background:#f5f5f5;">
    <tr><th>代码</th><th>名称</th><th>数量</th><th>成本价</th><th>现价</th>
        <th>市值</th><th>盈亏</th><th>仓位</th></tr>
    </thead><tbody>{rows}</tbody></table>"""


def _build_recent_trades(result: dict) -> str:
    trades = result.get("recent_trades", [])
    if not trades:
        return '<p style="font-size:13px;color:#888;">本期无交易</p>'

    rows = ""
    for t in trades:
        act_color = "#c23531" if t["action"] == "BUY" else "#2f4554"
        act_label = "买入" if t["action"] == "BUY" else "卖出"
        rows += f"""
        <tr>
            <td>{t['date']}</td>
            <td style="color:{act_color};font-weight:bold">{act_label}</td>
            <td>{t['symbol']}</td><td>{t['name']}</td>
            <td style="text-align:right">{t['price']:.3f}</td>
            <td style="text-align:right">{t['quantity']}</td>
        </tr>"""

    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:12px;" border="1" cellpadding="4">
    <thead style="background:#f5f5f5;">
    <tr><th>日期</th><th>方向</th><th>代码</th><th>名称</th><th>价格</th><th>数量</th></tr>
    </thead><tbody>{rows}</tbody></table>"""


def _build_reflection(result: dict) -> str:
    """根据回测数据自动生成策略反思"""
    m = result["metrics"]
    holdings = result.get("holdings", [])
    wins = result.get("top_wins", [])
    losses = result.get("top_losses", [])
    pool = _get_etf_pool()
    lines = []

    ret, sharpe, dd = m["total_return"], m["sharpe_ratio"], m["max_drawdown"]
    if ret > 0.15 and sharpe > 2:
        lines.append(f"本期表现优异 (收益 {ret:.2%}, Sharpe {sharpe:.2f})，策略在趋势行情中捕捉到了有效 alpha。")
    elif ret > 0:
        lines.append(f"本期收益 {ret:.2%}，风险调整后 Sharpe {sharpe:.2f}，表现中规中矩。")
    else:
        lines.append(f"本期收益 {ret:.2%}，策略处于回撤期，需关注风控熔断阈值。")

    if dd < -0.10:
        lines.append(f"最大回撤 {dd:.2%} 较深，需警惕连续亏损对资金曲线的侵蚀。")

    if wins:
        w = wins[0]
        name = pool.get(w["symbol"], w["symbol"])
        lines.append(f"最佳交易: {name} 盈利 ¥{w['pnl']:+,.0f} ({w['return_pct']:+.2%})，"
                     f"买入 {w['buy_price']:.3f} → 卖出 {w['sell_price']:.3f}。")

    if losses:
        l = losses[0]
        name = pool.get(l["symbol"], l["symbol"])
        lines.append(f"最差交易: {name} 亏损 ¥{l['pnl']:+,.0f} ({l['return_pct']:+.2%})，"
                     f"需反思入场时机是否追高。")

    if len(holdings) == 0:
        lines.append("当前空仓，等待下一次调仓信号。")
    else:
        conc = holdings[0]["weight"]
        if conc > 0.5:
            lines.append(f"持仓集中度偏高 ({holdings[0]['name']} 占 {conc:.0%})，注意个股风险。")

    return " ".join(lines)


def _build_strategy_section(result: dict, cid: str, section_num: int) -> str:
    m = result["metrics"]
    color = result["color"]
    return f"""
    <div style="margin-top:24px; border-left:4px solid {color}; padding-left:12px;">
        <h3 style="margin-bottom:4px;">策略 {section_num}: {result['label']}</h3>
        <p style="color:#666; font-size:13px; margin:4px 0 8px 0;">{result['desc']}</p>
        {_build_metrics_bar(m)}

        <details open style="margin-top:10px;">
        <summary style="font-weight:bold;font-size:13px;cursor:pointer;">当前持仓</summary>
        {_build_holdings_table(result)}
        </details>

        <details open style="margin-top:10px;">
        <summary style="font-weight:bold;font-size:13px;cursor:pointer;">最近交易</summary>
        {_build_recent_trades(result)}
        </details>

        <div style="margin-top:10px; padding:8px 12px; background:#f9f6f0; border-radius:4px; font-size:13px;">
            <b>策略反思:</b> {_build_reflection(result)}
        </div>

        <img src="cid:{cid}" style="width:100%; max-width:680px; border:1px solid #eee; margin-top:10px;" />
    </div>"""


def _build_html_report(top_scores: list, results: List[dict], report_date: str) -> str:
    ranking_table = _build_ranking_table(top_scores) if top_scores else "<p>暂无数据</p>"
    top_n = len(top_scores) if top_scores else 0

    strategy_sections = ""
    for i, r in enumerate(results, 1):
        cid = f"chart_{r['strategy_name'].lower()}"
        strategy_sections += _build_strategy_section(r, cid, i)

    best = max(results, key=lambda r: r["metrics"]["total_return"]) if results else None
    best_note = ""
    if best and len(results) > 1:
        best_note = (
            f'<p style="font-size:13px; color:#333; margin-top:12px;">'
            f'<b>本期最优:</b> {best["label"]}，'
            f'收益 {best["metrics"]["total_return"]:.2%}，'
            f'Sharpe {best["metrics"]["sharpe_ratio"]:.2f}</p>')

    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color:#333; max-width:740px; margin:auto; padding:16px;">

<h2 style="border-bottom:2px solid #c23531; padding-bottom:8px;">
    BreadFree 每日多策略决策报告
</h2>
<p style="color:#888; font-size:13px;">报告日期: {report_date} | 策略池: RotationStrategy / AgentStrategyV2 / EffiA</p>

<h3>Top {top_n} 标的因子排名 (全池量化评分)</h3>
{ranking_table}
<p style="font-size:12px; color:#999; margin-top:4px;">
效率分 = (动量 / 区间波动率) &times; R&sup2;&ensp;|&ensp;动量 = 20日ROC&ensp;|&ensp;R&sup2; 衡量趋势线性度
</p>

{'<h3>多策略收益对比</h3>' if len(results) > 1 else ''}
{'<img src="cid:chart_comparison" style="width:100%; max-width:700px; border:1px solid #eee;" />' if len(results) > 1 else ''}
{best_note}

{strategy_sections}

<hr style="margin-top:28px; border:none; border-top:1px solid #ddd;" />
<p style="font-size:11px; color:#aaa;">
此报告由 BreadFree 自动生成, 仅供研究参考, 不构成投资建议.
</p>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def generate_and_send_report():
    """生成并发送每日多策略报告"""
    now = datetime.now(TZ_SHANGHAI)
    report_date = now.strftime("%Y-%m-%d %H:%M")
    date_short = now.strftime("%m/%d")

    cfg = get_config().get("daily_report", {})
    top_n = cfg.get("top_n", 5)
    lookback = cfg.get("lookback_period", 20)
    backtest_days = cfg.get("backtest_days", 120)

    logger.info(f"[Report] 开始生成每日多策略报告 ({report_date})")

    # 1) Top-N 因子排名
    logger.info(f"[Report] 计算 Top-{top_n} 因子得分...")
    top_scores = calc_top_n_scores(top_n=top_n, lookback=lookback)
    if not top_scores:
        logger.warning("[Report] 无有效标的得分, 跳过发送")
        return False
    logger.info(f"[Report] Top-{top_n}: {[f'{s['symbol']}-{s['name']}' for s in top_scores]}")

    # 2) 运行 3 种策略回测
    strategy_order = ["RotationStrategy", "AgentStrategyV2", "EffiA"]
    results = []
    for name in strategy_order:
        r = _run_backtest(name, backtest_days, top_n=top_n, lookback=lookback)
        if r:
            results.append(r)

    if not results:
        logger.warning("[Report] 所有策略回测失败, 跳过发送")
        return False

    # 3) 生成图表
    images: Dict[str, bytes] = {}
    if len(results) > 1:
        logger.info("[Report] 生成多策略对比图...")
        images["chart_comparison"] = _generate_comparison_chart(results)

    for r in results:
        cid = f"chart_{r['strategy_name'].lower()}"
        logger.info(f"[Report] 生成 {r['strategy_name']} 收益曲线...")
        images[cid] = _generate_single_chart(r)

    # 4) 组装 HTML
    html = _build_html_report(top_scores, results, report_date)

    subject = (f"BreadFree 多策略报告 {date_short} | "
               + ", ".join(r["strategy_name"] for r in results))

    # 5) 发送
    ok = send_report_email(subject=subject, html_body=html, images=images)
    if ok:
        logger.info("[Report] 多策略报告发送成功")
    else:
        logger.error("[Report] 报告发送失败")
    return ok


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    generate_and_send_report()
