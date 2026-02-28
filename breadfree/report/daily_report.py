"""
每日策略决策报告生成器

功能:
    1. 计算全池 25 标的的最新多因子效率分, 输出 Top-N 决策表
    2. 运行 RotationStrategy 回测生成收益曲线 PNG
    3. 组装 HTML 报告, 通过 email_reporter 发送
"""

import io
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from breadfree.utils.config import get_config
from breadfree.utils.logger import get_logger
from breadfree.utils.email_reporter import send_report_email
from breadfree.utils.metrics import calculate_efficiency_metrics
from breadfree.data.data_fetcher import DataFetcher
from breadfree.data.database import get_db_manager
from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.effi_rotation_strategy import RotationStrategy

logger = get_logger(__name__)

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _get_etf_pool() -> dict:
    """返回 {symbol: name} 映射"""
    cfg = get_config()
    return dict(cfg.get("etf_pool", {"510300": "沪深300ETF"}))


def _fetch_latest_prices(symbols: list, lookback_days: int = 60) -> dict:
    """
    获取每只标的最近 N 天的收盘价序列.
    Returns: {symbol: [float, ...]}
    """
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
    """
    计算全池标的的因子得分, 返回 Top-N 列表.

    Returns:
        [{'rank': 1, 'symbol': '300124', 'name': '汇川技术',
          'momentum': 0.17, 'volatility': 0.012, 'r2': 0.91,
          'efficiency': 3.07, 'close': 69.0}, ...]
    """
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


def _generate_equity_curve_png(backtest_days: int = 120) -> bytes:
    """运行 RotationStrategy 回测, 返回权益曲线 PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    now = datetime.now(TZ_SHANGHAI)
    end_date = now.strftime("%Y%m%d")
    start_date = (now - timedelta(days=backtest_days)).strftime("%Y%m%d")

    cfg = get_config()
    symbols = list(cfg.get("etf_pool", {"510300": "沪深300ETF"}).keys())
    initial_cash = cfg.get("initial_cash", 100000.0)

    engine = BacktestEngine(
        strategy_cls=RotationStrategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        lookback_period=cfg.get("daily_report", {}).get("lookback_period", 20),
        hold_period=20,
        top_n=cfg.get("daily_report", {}).get("top_n", 5),
    )
    engine.run()

    if not engine.broker.equity_curve:
        logger.warning("[Report] 回测无权益数据, 生成空图")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    dates = [d["date"] for d in engine.broker.equity_curve]
    equities = [d["equity"] for d in engine.broker.equity_curve]
    total_ret = (equities[-1] / initial_cash - 1) * 100

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, equities, color="#c23531", linewidth=1.8, label="Strategy Equity")
    ax.fill_between(dates, initial_cash, equities, alpha=0.08, color="#c23531")
    ax.axhline(y=initial_cash, color="#999", linestyle="--", linewidth=0.8)

    ax.set_title(f"RotationStrategy ({start_date}~{end_date})  Return: {total_ret:+.2f}%",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity (CNY)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _build_html_report(top_scores: list, report_date: str) -> str:
    """组装 HTML 邮件正文"""

    rows_html = ""
    for item in top_scores:
        mom_pct = f"{item['momentum'] * 100:+.2f}%"
        vol_pct = f"{item['volatility'] * 100:.2f}%"
        r2_val = f"{item['r2']:.2f}"
        eff_val = f"{item['efficiency']:.2f}"
        close_val = f"{item['close']:.3f}"

        if item["efficiency"] >= 2.0:
            signal = '<span style="color:#c23531;font-weight:bold">&#9650; 强势</span>'
        elif item["efficiency"] >= 1.0:
            signal = '<span style="color:#d48806;font-weight:bold">&#9650; 看多</span>'
        elif item["efficiency"] >= 0:
            signal = '<span style="color:#666">&#9644; 中性</span>'
        else:
            signal = '<span style="color:#389e0d">&#9660; 看空</span>'

        rows_html += f"""
        <tr>
            <td style="text-align:center">{item['rank']}</td>
            <td>{item['symbol']}</td>
            <td>{item['name']}</td>
            <td style="text-align:right">{close_val}</td>
            <td style="text-align:right">{mom_pct}</td>
            <td style="text-align:right">{vol_pct}</td>
            <td style="text-align:right">{r2_val}</td>
            <td style="text-align:right;font-weight:bold">{eff_val}</td>
            <td style="text-align:center">{signal}</td>
        </tr>"""

    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color:#333; max-width:720px; margin:auto; padding:16px;">

<h2 style="border-bottom:2px solid #c23531; padding-bottom:8px;">
    BreadFree 每日决策报告
</h2>
<p style="color:#888; font-size:13px;">报告日期: {report_date} | 策略: RotationStrategy (效率轮动)</p>

<h3>Top {len(top_scores)} 标的因子排名</h3>
<table style="border-collapse:collapse; width:100%; font-size:13px;" border="1" cellpadding="6">
<thead style="background:#f5f5f5;">
<tr>
    <th>排名</th><th>代码</th><th>名称</th><th>最新价</th>
    <th>动量</th><th>波动率</th><th>R&sup2;</th><th>效率分</th><th>信号</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<p style="font-size:12px; color:#999; margin-top:4px;">
效率分 = (动量 / 区间波动率) &times; R&sup2;&ensp;|&ensp;
动量 = {len(top_scores) and top_scores[0].get('_lookback', 20) or 20}日ROC&ensp;|&ensp;
R&sup2; 衡量趋势线性度
</p>

<h3>投资组合收益曲线</h3>
<img src="cid:equity_curve" style="width:100%; max-width:700px; border:1px solid #eee;" alt="equity curve" />

<hr style="margin-top:24px; border:none; border-top:1px solid #ddd;" />
<p style="font-size:11px; color:#aaa;">
此报告由 BreadFree 自动生成, 仅供研究参考, 不构成投资建议.
</p>
</body>
</html>"""
    return html


def generate_and_send_report():
    """生成并发送每日报告 (主入口)"""
    now = datetime.now(TZ_SHANGHAI)
    report_date = now.strftime("%Y-%m-%d %H:%M")
    date_short = now.strftime("%m/%d")

    cfg = get_config().get("daily_report", {})
    top_n = cfg.get("top_n", 5)
    lookback = cfg.get("lookback_period", 20)
    backtest_days = cfg.get("backtest_days", 120)

    logger.info(f"[Report] 开始生成每日报告 ({report_date})")

    logger.info(f"[Report] 计算 Top-{top_n} 因子得分...")
    top_scores = calc_top_n_scores(top_n=top_n, lookback=lookback)
    if not top_scores:
        logger.warning("[Report] 无有效标的得分, 跳过发送")
        return False

    logger.info(f"[Report] Top-{top_n}: {[f'{s['symbol']}-{s['name']}' for s in top_scores]}")

    logger.info(f"[Report] 运行 {backtest_days} 天回测生成收益曲线...")
    get_db_manager().clear_cache()
    equity_png = _generate_equity_curve_png(backtest_days=backtest_days)

    html = _build_html_report(top_scores, report_date)
    subject = f"BreadFree 决策报告 {date_short} | Top-{top_n}: " + ", ".join(
        f"{s['name']}" for s in top_scores[:3]
    ) + "..."

    ok = send_report_email(
        subject=subject,
        html_body=html,
        images={"equity_curve": equity_png},
    )

    if ok:
        logger.info("[Report] 报告发送成功")
    else:
        logger.error("[Report] 报告发送失败")
    return ok


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    generate_and_send_report()
