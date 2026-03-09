"""
DailySnapshotWriter — 选股择时数据持久化封装

职责:
    1. 将每日因子快照（全池 25 标的）写入 daily_factors 表
    2. 将全市场发现扫描结果写入 discovery_scan 表
    3. 将当日选股决策写入 rebalance_log 表
    4. 导出 JSON 文件到 breadfree/data/cache/snapshot/ 供下游消费

JSON 文件格式:
    breadfree/data/cache/snapshot/YYYYMMDD.json
    breadfree/data/cache/snapshot/latest.json  (软链接式覆盖，始终指向最新)

供下游消费示例:
    import json
    data = json.load(open("breadfree/data/cache/snapshot/latest.json"))
    top5 = data["selection"]          # 今日 Top-5 选股
    all_factors = data["all_factors"] # 全池 25 标的因子排名
    discovery = data["discovery"]     # 全市场发现新标的
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "cache", "snapshot")
os.makedirs(_SNAPSHOT_DIR, exist_ok=True)


class DailySnapshotWriter:
    """
    选股择时数据持久化封装

    使用方式:
        writer = DailySnapshotWriter()
        writer.write(
            trade_date="20260309",
            all_scores=all_scores,    # 全池因子列表（含 pool_rank / is_selected）
            top_n=5,
            lookback=20,
            discovery_summary=discovery_summary,  # StockDiscovery.get_discovery_summary() 返回值
        )
        writer.export_json("20260309")
    """

    def __init__(self, db_path: str = "breadfree.db"):
        from .database import get_db_manager
        self.db = get_db_manager(db_path)

    # ──────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────

    def write(
        self,
        trade_date: str,
        all_scores: List[dict],
        top_n: int,
        lookback: int,
        discovery_summary: Optional[dict] = None,
    ) -> dict:
        """
        一次性持久化当日选股择时数据。

        Args:
            trade_date:        数据截止日期，YYYYMMDD
            all_scores:        全池因子列表，由 calc_all_factor_scores() 返回，
                               须含 pool_rank / is_selected 字段
            top_n:             Top-N 配置
            lookback:          回看期
            discovery_summary: StockDiscovery.get_discovery_summary() 返回值，可为 None

        Returns:
            {factors_saved, discovery_saved, rebalance_saved} 写入计数
        """
        result = {"factors_saved": 0, "discovery_saved": 0, "rebalance_saved": False}

        # 1) 写因子快照
        if all_scores:
            factor_records = [
                {**s, "top_n": top_n, "lookback_period": lookback}
                for s in all_scores
            ]
            result["factors_saved"] = self.db.save_factor_snapshot(factor_records)

        # 2) 写发现扫描结果
        if discovery_summary:
            discoveries = discovery_summary.get("top_discoveries", [])
            is_stale = discovery_summary.get("is_stale_cache", False)
            scan_date = discovery_summary.get("scan_time", trade_date)
            # scan_time 格式 "2026-02-28 07:41" → 取日期部分
            if isinstance(scan_date, str) and len(scan_date) >= 10:
                scan_date_str = scan_date[:10].replace("-", "")
            else:
                scan_date_str = trade_date
            if discoveries:
                result["discovery_saved"] = self.db.save_discovery_scan(
                    discoveries, scan_date_str, is_stale_cache=is_stale
                )

        # 3) 写调仓记录（选股结果 → 视为每日 snapshot 触发的调仓建议）
        selected = [s for s in all_scores if s.get("is_selected")]
        if selected:
            rebalance_record = {
                "trade_date": trade_date,
                "strategy": "daily_snapshot",
                "trigger_type": "daily_snapshot",
                "top_n": top_n,
                "lookback_period": lookback,
                "selected": [
                    {
                        "rank": s.get("pool_rank"),
                        "symbol": s["symbol"],
                        "name": s.get("name", ""),
                        "efficiency": round(float(s.get("efficiency", 0) or 0), 4),
                        "momentum": round(float(s.get("momentum", 0) or 0), 4),
                        "weight": round(1.0 / top_n, 4),
                    }
                    for s in selected
                ],
            }
            try:
                self.db.save_rebalance_log(rebalance_record)
                result["rebalance_saved"] = True
            except Exception as e:
                logger.warning(f"[Snapshot] 调仓记录写入失败: {e}")

        # 4) 导出 JSON
        self.export_json(trade_date, all_scores, top_n, lookback, discovery_summary)

        logger.info(
            f"[Snapshot] {trade_date} 持久化完成: "
            f"因子 {result['factors_saved']} 条, "
            f"发现 {result['discovery_saved']} 条, "
            f"调仓记录 {'✓' if result['rebalance_saved'] else '✗'}"
        )
        return result

    # ──────────────────────────────────────────────────────────
    # JSON 导出
    # ──────────────────────────────────────────────────────────

    def export_json(
        self,
        trade_date: str,
        all_scores: List[dict],
        top_n: int,
        lookback: int,
        discovery_summary: Optional[dict] = None,
        out_dir: str = None,
    ) -> str:
        """
        导出当日快照为 JSON 文件。

        写入两个文件:
            {out_dir}/YYYYMMDD.json   — 按日期归档
            {out_dir}/latest.json     — 始终为最新快照（覆盖写入）

        Returns:
            写入的日期文件路径
        """
        out_dir = out_dir or _SNAPSHOT_DIR

        # 格式化日期
        d = trade_date.replace("-", "")  # 统一为 YYYYMMDD
        date_label = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        payload = {
            "trade_date": date_label,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "top_n": top_n,
            "lookback_period": lookback,
            "selection": [
                _to_json_safe(s)
                for s in all_scores
                if s.get("is_selected")
            ],
            "all_factors": [_to_json_safe(s) for s in all_scores],
            "discovery": _build_discovery_payload(discovery_summary),
        }

        dated_path = os.path.join(out_dir, f"{d}.json")
        latest_path = os.path.join(out_dir, "latest.json")

        for path in (dated_path, latest_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[Snapshot] JSON 导出: {dated_path}")
        return dated_path

    # ──────────────────────────────────────────────────────────
    # 查询助手
    # ──────────────────────────────────────────────────────────

    def get_latest_selection(self, top_n: int = None) -> List[dict]:
        """查询最新 Top-N 选股（来自 daily_factors）"""
        return self.db.get_latest_factors(top_n=top_n)

    def get_latest_discovery(self) -> List[dict]:
        """查询最新发现扫描结果"""
        return self.db.get_latest_discovery()

    def get_rebalance_history(self, days: int = 30) -> List[dict]:
        """查询最近 N 天的调仓记录"""
        return self.db.get_rebalance_log(days=days, strategy="daily_snapshot")


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def _to_json_safe(d: dict) -> dict:
    """将 numpy 数值转为 Python 原生类型，供 JSON 序列化"""
    import numpy as np
    result = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = v.item()
        elif isinstance(v, float) and (v != v):  # NaN
            result[k] = None
        else:
            result[k] = v
    return result


def _build_discovery_payload(discovery_summary: Optional[dict]) -> dict:
    """将 discovery_summary 格式化为 JSON 友好结构"""
    if not discovery_summary:
        return {"total_discovered": 0, "top_discoveries": [], "is_stale_cache": False}

    discoveries = discovery_summary.get("top_discoveries", [])
    return {
        "total_discovered": discovery_summary.get("total_discovered", len(discoveries)),
        "avg_efficiency": round(float(discovery_summary.get("avg_efficiency", 0) or 0), 4),
        "max_efficiency": round(float(discovery_summary.get("max_efficiency", 0) or 0), 4),
        "scan_time": discovery_summary.get("scan_time", ""),
        "is_stale_cache": discovery_summary.get("is_stale_cache", False),
        "stale_cache_time": discovery_summary.get("stale_cache_time"),
        "top_discoveries": [_to_json_safe(d) for d in discoveries],
    }


# ──────────────────────────────────────────────────────────────
# 独立运行入口（供 cron 直接调用）
# ──────────────────────────────────────────────────────────────

def run_snapshot(trade_date: str = None) -> dict:
    """
    独立计算并持久化当日选股择时快照。

    无需发邮件，只做数据计算 + 存库 + 导出 JSON。
    供 daily_snapshot.py 或其他 cron 直接调用。
    """
    import sys
    import os
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from ..utils.config import get_config
    from ..utils.metrics import calculate_efficiency_metrics
    from .data_fetcher import DataFetcher
    from .database import get_db_manager
    from .stock_discovery import StockDiscovery
    from datetime import datetime, timedelta

    if trade_date is None:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        d = now.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        trade_date = d.strftime("%Y%m%d")

    logger.info(f"[Snapshot] 开始计算 {trade_date} 快照...")

    cfg = get_config()
    pool: dict = cfg.get("etf_pool", {})
    top_n: int = cfg.get("daily_report", {}).get("top_n", 5)
    lookback: int = cfg.get("daily_report", {}).get("lookback_period", 20)

    symbols = list(pool.keys())

    # ── 刷新行情数据 ──────────────────────────────────────────
    lookback_days = lookback * 3
    end_dt = datetime.strptime(trade_date, "%Y%m%d")
    start_date = (end_dt - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    import pandas as pd
    fetcher = DataFetcher(
        data_dir=os.path.join(os.path.dirname(__file__), "cache"),
        data_source="akshare",
    )
    db = get_db_manager()
    db.clear_cache()

    # 确保各标的有到 trade_date 的最新数据（与 daily_report._refresh_pool_data_for_report 一致）
    logger.info("[Snapshot] 刷新池内行情至最近交易日...")
    end_pd = pd.to_datetime(trade_date)
    for i, symbol in enumerate(symbols):
        df = db.get_daily_data(symbol, start_date, trade_date)
        if df.empty or (not df.empty and df.index.max() < end_pd - timedelta(days=5)):
            fetcher.fetch_a_stock_daily(symbol, start_date, trade_date)
        if (i + 1) % 10 == 0:
            logger.info(f"[Snapshot] 刷新行情 {i + 1}/{len(symbols)}...")

    db.preload_symbols(symbols, start_date, trade_date)

    # ── 计算全池因子 ──────────────────────────────────────────
    all_scores = _compute_all_factors(
        symbols, pool, db, fetcher, start_date, trade_date, lookback, top_n
    )

    if not all_scores:
        logger.warning("[Snapshot] 无有效标的，快照终止")
        return {"error": "no_data"}

    # ── 全市场发现扫描 ────────────────────────────────────────
    discovery_summary = None
    disc_cfg = cfg.get("discovery", {})
    if disc_cfg.get("enabled", True):
        try:
            disc = StockDiscovery(
                min_amount=float(disc_cfg.get("min_amount", 5e7)),
                min_circ_mv=float(disc_cfg.get("min_circ_mv", 5e9)),
                efficiency_threshold=float(disc_cfg.get("efficiency_threshold", 0.5)),
                lookback_period=lookback,
                max_discover=int(disc_cfg.get("max_discover", 30)),
            )
            disc.get_expanded_pool(end_date=trade_date, max_expand=10)
            discovery_summary = disc.get_discovery_summary(trade_date)
        except Exception as e:
            logger.warning(f"[Snapshot] 发现扫描失败: {e}")

    # ── 持久化 ────────────────────────────────────────────────
    writer = DailySnapshotWriter()
    result = writer.write(trade_date, all_scores, top_n, lookback, discovery_summary)
    result["trade_date"] = trade_date
    return result


def _compute_all_factors(
    symbols: List[str],
    pool: dict,
    db,
    fetcher,
    start_date: str,
    end_date: str,
    lookback: int,
    top_n: int,
) -> List[dict]:
    """计算全池所有标的的因子分并加上排名/选股标志"""
    import pandas as pd
    from ..utils.metrics import calculate_efficiency_metrics

    scored = []
    for symbol in symbols:
        # 三层取数：内存 → DB → 远端
        df = db.get_preloaded(symbol)
        if df is None or df.empty:
            df = db.get_daily_data(symbol, start_date, end_date)
        if df is None or df.empty:
            df = fetcher.fetch_a_stock_daily(symbol, start_date, end_date)
        if df is None or df.empty:
            continue

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df.set_index("trade_date", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        closes = df["close"].dropna().tolist()
        if len(closes) < lookback:
            continue

        metrics = calculate_efficiency_metrics(closes, lookback)
        if metrics is None:
            continue

        scored.append({
            "trade_date": end_date,
            "symbol": symbol,
            "name": pool.get(symbol, symbol),
            "close": metrics["close"],
            "momentum": metrics["momentum"],
            "volatility": metrics["volatility"],
            "r2": metrics["r2"],
            "efficiency": metrics["efficiency"],
        })

    if not scored:
        return []

    # 按效率分倒序排名
    scored.sort(key=lambda x: x["efficiency"], reverse=True)
    selected_symbols = {s["symbol"] for s in scored[:top_n]}
    for rank, s in enumerate(scored, 1):
        s["pool_rank"] = rank
        s["is_selected"] = s["symbol"] in selected_symbols

    return scored
