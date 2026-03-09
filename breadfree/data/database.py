"""
DatabaseManager — 三层存储引擎

Layer 1  Memory Cache  numpy 数组预加载, 回测期间零 IO
Layer 2  SQLite DB     持久化, 覆盖索引, 增量写入
Layer 3  Remote API    DataFetcher 降级链 (东方财富/腾讯/新浪/AkShare)

高频量化借鉴:
    - 回测前一次性 bulk load 全部数据到 numpy (连续内存, L1 cache 友好)
    - SQLite WAL 模式 + 批量 INSERT OR IGNORE (写入吞吐 ~100k rows/s)
    - 覆盖索引避免回表 (日线查询只走 ix_daily_cover)
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .db_models import (
    Base, StockInfo, DailyData, MonthlyData,
    TechnicalIndicators, NewsArticle, MarketIntelDaily,
    SectorInfo, StockSectorMapping, SentimentData,
    DailyFactors, DiscoveryScan, RebalanceLog,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """三层存储管理器"""

    def __init__(self, db_path: str = "breadfree.db"):
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.Session = sessionmaker(bind=self.engine)

        # 启用 WAL 模式 (写入性能提升 ~5x)
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB cache
            conn.commit()

        Base.metadata.create_all(self.engine)

        # Layer 1: 内存缓存 {symbol: DataFrame}
        self._mem_cache: Dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────────
    # Layer 2: 日线数据 CRUD
    # ──────────────────────────────────────────────────────────

    def get_daily_data(self, symbol: str, start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """查询日线数据 — 先查内存, 再查 DB"""
        # Layer 1: 内存缓存
        if symbol in self._mem_cache:
            df = self._mem_cache[symbol]
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if not df.empty:
                return df

        # Layer 2: SQLite
        session = self.Session()
        try:
            q = session.query(DailyData).filter(DailyData.symbol == symbol)
            if start_date:
                q = q.filter(DailyData.trade_date >= pd.to_datetime(start_date).date())
            if end_date:
                q = q.filter(DailyData.trade_date <= pd.to_datetime(end_date).date())
            q = q.order_by(DailyData.trade_date)
            df = pd.read_sql(q.statement, session.bind)
            if not df.empty and "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
            return df
        finally:
            session.close()

    def get_latest_date(self, symbol: str) -> Optional[date]:
        """获取某标的最新入库日期"""
        session = self.Session()
        try:
            row = (session.query(DailyData.trade_date)
                   .filter(DailyData.symbol == symbol)
                   .order_by(DailyData.trade_date.desc())
                   .first())
            return row[0] if row else None
        finally:
            session.close()

    def bulk_upsert_daily(self, symbol: str, df: pd.DataFrame):
        """批量写入日线数据 (INSERT OR IGNORE, 幂等)"""
        if df.empty:
            return 0

        work = df.reset_index() if df.index.name in ("date", "trade_date") else df.copy()
        date_col = "trade_date" if "trade_date" in work.columns else "date"
        if date_col not in work.columns:
            logger.warning(f"No date column for {symbol}")
            return 0

        work[date_col] = pd.to_datetime(work[date_col]).dt.date

        rows = []
        for _, r in work.iterrows():
            rows.append({
                "symbol": symbol,
                "trade_date": r[date_col],
                "open": r.get("open"), "close": r.get("close"),
                "high": r.get("high"), "low": r.get("low"),
                "volume": r.get("volume"), "amount": r.get("amount"),
                "amplitude": r.get("amplitude"), "pct_chg": r.get("pct_chg"),
                "change": r.get("change"), "turnover": r.get("turnover"),
            })

        if not rows:
            return 0

        # SQLite INSERT OR IGNORE (幂等, 重复跳过)
        sql = text("""
            INSERT OR IGNORE INTO daily_data
                (symbol, trade_date, open, close, high, low, volume, amount,
                 amplitude, pct_chg, change, turnover)
            VALUES
                (:symbol, :trade_date, :open, :close, :high, :low, :volume, :amount,
                 :amplitude, :pct_chg, :change, :turnover)
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, rows)

        # 更新内存缓存
        if symbol in self._mem_cache:
            del self._mem_cache[symbol]

        return len(rows)

    # ──────────────────────────────────────────────────────────
    # Layer 1: 内存预加载 (高频量化核心)
    # ──────────────────────────────────────────────────────────

    def preload_symbols(self, symbols: List[str], start_date: str, end_date: str):
        """一次性将多只标的的日线数据加载到内存 — 回测前调用"""
        t0 = datetime.now()
        total = 0
        for sym in symbols:
            df = self.get_daily_data(sym, start_date, end_date)
            if not df.empty:
                self._mem_cache[sym] = df
                total += len(df)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"[DB] Preloaded {total} rows for {len(symbols)} symbols in {elapsed:.2f}s")

    def get_preloaded(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取内存中预加载的数据"""
        return self._mem_cache.get(symbol)

    def clear_cache(self):
        """清空内存缓存"""
        self._mem_cache.clear()

    # ──────────────────────────────────────────────────────────
    # 新闻持久化
    # ──────────────────────────────────────────────────────────

    def store_news(self, symbol: str, articles: List[dict]):
        """批量存储新闻文章 (去重)"""
        if not articles:
            return 0

        sql = text("""
            INSERT OR IGNORE INTO news_articles
                (symbol, publish_date, title, content, source, url)
            VALUES
                (:symbol, :publish_date, :title, :content, :source, :url)
        """)

        rows = []
        for a in articles:
            pub = pd.to_datetime(a.get("publish_date") or a.get("发布时间"))
            rows.append({
                "symbol": symbol,
                "publish_date": pub.to_pydatetime() if hasattr(pub, "to_pydatetime") else pub,
                "title": a.get("title") or a.get("新闻标题", ""),
                "content": a.get("content") or a.get("新闻内容", ""),
                "source": a.get("source") or a.get("文章来源", ""),
                "url": a.get("url") or a.get("新闻链接", ""),
            })

        with self.engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)

    def get_news(self, symbol: str, start_date: str = None,
                 end_date: str = None, limit: int = 50) -> pd.DataFrame:
        """查询新闻"""
        session = self.Session()
        try:
            q = session.query(NewsArticle).filter(NewsArticle.symbol == symbol)
            if start_date:
                q = q.filter(NewsArticle.publish_date >= pd.to_datetime(start_date))
            if end_date:
                q = q.filter(NewsArticle.publish_date <= pd.to_datetime(end_date))
            q = q.order_by(NewsArticle.publish_date.desc()).limit(limit)
            return pd.read_sql(q.statement, session.bind)
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────
    # 市场情报持久化
    # ──────────────────────────────────────────────────────────

    def store_market_intel(self, intel_type: str, date_val, data: dict,
                           symbol: str = None):
        """存储单条市场情报"""
        sql = text("""
            INSERT OR IGNORE INTO market_intel_daily
                (intel_type, symbol, date, data_json)
            VALUES
                (:intel_type, :symbol, :date, :data_json)
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {
                "intel_type": intel_type,
                "symbol": symbol or "",
                "date": pd.to_datetime(date_val).date() if date_val else None,
                "data_json": json.dumps(data, ensure_ascii=False, default=str),
            })

    def get_market_intel(self, intel_type: str, date_val,
                         symbol: str = None) -> Optional[dict]:
        """查询市场情报"""
        session = self.Session()
        try:
            q = (session.query(MarketIntelDaily)
                 .filter(MarketIntelDaily.intel_type == intel_type,
                         MarketIntelDaily.date == pd.to_datetime(date_val).date()))
            if symbol:
                q = q.filter(MarketIntelDaily.symbol == symbol)
            else:
                q = q.filter(MarketIntelDaily.symbol == "")
            row = q.first()
            if row:
                return json.loads(row.data_json)
            return None
        finally:
            session.close()

    # ──────────────────────────────────────────────────────────
    # 选股择时数据持久化
    # ──────────────────────────────────────────────────────────

    def save_factor_snapshot(self, records: List[dict]) -> int:
        """
        批量保存每日全池因子快照到 daily_factors 表（INSERT OR REPLACE）。

        Args:
            records: list of dict，每条需包含 trade_date, symbol，其余字段可选。
        Returns:
            实际写入行数。
        """
        if not records:
            return 0

        sql = text("""
            INSERT OR REPLACE INTO daily_factors
                (trade_date, symbol, name, close,
                 momentum, volatility, r2, efficiency,
                 accel, composite, drawdown_from_high,
                 pool_rank, is_selected, top_n, lookback_period, created_at)
            VALUES
                (:trade_date, :symbol, :name, :close,
                 :momentum, :volatility, :r2, :efficiency,
                 :accel, :composite, :drawdown_from_high,
                 :pool_rank, :is_selected, :top_n, :lookback_period, :created_at)
        """)

        rows = []
        for r in records:
            td = r.get("trade_date")
            if td and not isinstance(td, date):
                td = pd.to_datetime(td).date()
            rows.append({
                "trade_date": td,
                "symbol": r.get("symbol", ""),
                "name": r.get("name", ""),
                "close": r.get("close"),
                "momentum": r.get("momentum"),
                "volatility": r.get("volatility"),
                "r2": r.get("r2"),
                "efficiency": r.get("efficiency"),
                "accel": r.get("accel"),
                "composite": r.get("composite"),
                "drawdown_from_high": r.get("drawdown_from_high"),
                "pool_rank": r.get("pool_rank"),
                "is_selected": int(bool(r.get("is_selected", False))),
                "top_n": r.get("top_n"),
                "lookback_period": r.get("lookback_period"),
                "created_at": datetime.now(),
            })

        with self.engine.begin() as conn:
            conn.execute(sql, rows)

        logger.info(f"[DB] 保存因子快照 {len(rows)} 条 (trade_date={rows[0]['trade_date']})")
        return len(rows)

    def save_discovery_scan(self, records: List[dict], scan_date: str,
                            is_stale_cache: bool = False) -> int:
        """
        批量保存全市场发现扫描结果到 discovery_scan 表（INSERT OR REPLACE）。

        Args:
            records: list of dict，来自 StockDiscovery.discover() 返回值。
            scan_date: 扫描日期 YYYYMMDD。
            is_stale_cache: 数据是否来自过期缓存。
        Returns:
            实际写入行数。
        """
        if not records:
            return 0

        sql = text("""
            INSERT OR REPLACE INTO discovery_scan
                (scan_date, symbol, name, asset_type,
                 latest_price, circ_mv, amount, turnover_rate,
                 momentum, volatility, r2, efficiency,
                 scan_rank, is_stale_cache, created_at)
            VALUES
                (:scan_date, :symbol, :name, :asset_type,
                 :latest_price, :circ_mv, :amount, :turnover_rate,
                 :momentum, :volatility, :r2, :efficiency,
                 :scan_rank, :is_stale_cache, :created_at)
        """)

        sd = pd.to_datetime(scan_date).date() if scan_date else date.today()
        rows = []
        for rank, r in enumerate(records, 1):
            rows.append({
                "scan_date": sd,
                "symbol": r.get("symbol", ""),
                "name": r.get("name", ""),
                "asset_type": r.get("asset_type", "stock"),
                "latest_price": r.get("latest_price"),
                "circ_mv": r.get("circ_mv"),
                "amount": r.get("amount"),
                "turnover_rate": r.get("turnover_rate"),
                "momentum": r.get("momentum"),
                "volatility": r.get("volatility"),
                "r2": r.get("r2"),
                "efficiency": r.get("efficiency"),
                "scan_rank": r.get("scan_rank", rank),
                "is_stale_cache": int(is_stale_cache),
                "created_at": datetime.now(),
            })

        with self.engine.begin() as conn:
            conn.execute(sql, rows)

        logger.info(f"[DB] 保存发现扫描 {len(rows)} 条 (scan_date={sd}, stale={is_stale_cache})")
        return len(rows)

    def save_rebalance_log(self, record: dict) -> None:
        """
        保存调仓/择时决策记录（INSERT OR REPLACE）。

        Args:
            record: dict，需包含 trade_date, strategy；其余字段可选。
        """
        sql = text("""
            INSERT OR REPLACE INTO rebalance_log
                (trade_date, strategy, trigger_type, trigger_score,
                 top_n, lookback_period, selected_json, signal_json, created_at)
            VALUES
                (:trade_date, :strategy, :trigger_type, :trigger_score,
                 :top_n, :lookback_period, :selected_json, :signal_json, :created_at)
        """)

        td = record.get("trade_date")
        if td and not isinstance(td, date):
            td = pd.to_datetime(td).date()

        selected = record.get("selected_json") or record.get("selected", [])
        signal = record.get("signal_json") or record.get("signal", {})

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "trade_date": td,
                "strategy": record.get("strategy", "daily_snapshot"),
                "trigger_type": record.get("trigger_type", "daily_snapshot"),
                "trigger_score": record.get("trigger_score"),
                "top_n": record.get("top_n"),
                "lookback_period": record.get("lookback_period"),
                "selected_json": json.dumps(selected, ensure_ascii=False, default=str)
                    if not isinstance(selected, str) else selected,
                "signal_json": json.dumps(signal, ensure_ascii=False, default=str)
                    if not isinstance(signal, str) else signal,
                "created_at": datetime.now(),
            })

        logger.info(f"[DB] 保存调仓记录 strategy={record.get('strategy')} date={td}")

    def get_latest_factors(self, top_n: int = None,
                           trade_date: str = None) -> List[dict]:
        """
        查询最新（或指定日期）全池因子快照，按 pool_rank 排序。

        Args:
            top_n: 若指定，只返回 pool_rank <= top_n 的标的（入选标的）。
            trade_date: 若指定，查询该日期；否则查询最新日期。
        Returns:
            list of dict。
        """
        with self.engine.connect() as conn:
            if trade_date:
                td = pd.to_datetime(trade_date).date()
            else:
                row = conn.execute(
                    text("SELECT MAX(trade_date) FROM daily_factors")
                ).fetchone()
                td = row[0] if row and row[0] else None
                if not td:
                    return []

            q = "SELECT * FROM daily_factors WHERE trade_date = :td"
            params: dict = {"td": td}
            if top_n:
                q += " AND is_selected = 1"
            q += " ORDER BY pool_rank"

            rows = conn.execute(text(q), params).mappings().all()
            result = [dict(r) for r in rows]

        if top_n:
            result = result[:top_n]
        return result

    def get_factor_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """查询某标的近 N 天因子历史"""
        sql = text("""
            SELECT trade_date, symbol, name, close, momentum, volatility, r2,
                   efficiency, pool_rank, is_selected
            FROM daily_factors
            WHERE symbol = :symbol
            ORDER BY trade_date DESC
            LIMIT :days
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"symbol": symbol, "days": days})
        return df.sort_values("trade_date") if not df.empty else df

    def get_latest_discovery(self, scan_date: str = None) -> List[dict]:
        """查询最新（或指定日期）发现扫描结果，按 scan_rank 排序"""
        with self.engine.connect() as conn:
            if scan_date:
                sd = pd.to_datetime(scan_date).date()
            else:
                row = conn.execute(
                    text("SELECT MAX(scan_date) FROM discovery_scan")
                ).fetchone()
                sd = row[0] if row and row[0] else None
                if not sd:
                    return []

            rows = conn.execute(
                text("SELECT * FROM discovery_scan WHERE scan_date = :sd ORDER BY scan_rank"),
                {"sd": sd},
            ).mappings().all()
            return [dict(r) for r in rows]

    def get_rebalance_log(self, days: int = 30,
                          strategy: str = None) -> List[dict]:
        """查询近 N 天调仓记录"""
        sql_parts = ["SELECT * FROM rebalance_log WHERE 1=1"]
        params: dict = {}
        if strategy:
            sql_parts.append("AND strategy = :strategy")
            params["strategy"] = strategy
        sql_parts.append("ORDER BY trade_date DESC LIMIT :days")
        params["days"] = days

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(" ".join(sql_parts)), params
            ).mappings().all()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────────
    # 选股择时数据持久化
    # ──────────────────────────────────────────────────────────

    def save_factor_snapshot(self, records: List[dict]) -> int:
        """
        批量写入每日因子快照（INSERT OR REPLACE，幂等）。

        records 每条须含: trade_date, symbol, name, close,
            momentum, volatility, r2, efficiency,
            pool_rank, is_selected, top_n, lookback_period
        可选: accel, composite, drawdown_from_high
        """
        if not records:
            return 0

        sql = text("""
            INSERT OR REPLACE INTO daily_factors
                (trade_date, symbol, name, close,
                 momentum, volatility, r2, efficiency,
                 accel, composite, drawdown_from_high,
                 pool_rank, is_selected, top_n, lookback_period, created_at)
            VALUES
                (:trade_date, :symbol, :name, :close,
                 :momentum, :volatility, :r2, :efficiency,
                 :accel, :composite, :drawdown_from_high,
                 :pool_rank, :is_selected, :top_n, :lookback_period, :created_at)
        """)

        rows = []
        now = datetime.now()
        for r in records:
            rows.append({
                "trade_date": pd.to_datetime(r["trade_date"]).date(),
                "symbol": r["symbol"],
                "name": r.get("name", ""),
                "close": r.get("close"),
                "momentum": r.get("momentum"),
                "volatility": r.get("volatility"),
                "r2": r.get("r2"),
                "efficiency": r.get("efficiency"),
                "accel": r.get("accel"),
                "composite": r.get("composite"),
                "drawdown_from_high": r.get("drawdown_from_high"),
                "pool_rank": r.get("pool_rank"),
                "is_selected": int(bool(r.get("is_selected", False))),
                "top_n": r.get("top_n"),
                "lookback_period": r.get("lookback_period"),
                "created_at": now,
            })

        with self.engine.begin() as conn:
            conn.execute(sql, rows)
        logger.info(f"[DB] 写入 {len(rows)} 条因子快照 (trade_date={rows[0]['trade_date']})")
        return len(rows)

    def save_discovery_scan(self, records: List[dict], scan_date: str,
                            is_stale_cache: bool = False) -> int:
        """
        批量写入全市场发现扫描结果（INSERT OR REPLACE，幂等）。

        records 为 StockDiscovery.discover() 的返回值列表，
        每条须含: symbol, name, asset_type, latest_price, circ_mv,
            amount, turnover_rate, momentum, volatility, r2, efficiency
        """
        if not records:
            return 0

        sql = text("""
            INSERT OR REPLACE INTO discovery_scan
                (scan_date, symbol, name, asset_type,
                 latest_price, circ_mv, amount, turnover_rate,
                 momentum, volatility, r2, efficiency,
                 scan_rank, is_stale_cache, created_at)
            VALUES
                (:scan_date, :symbol, :name, :asset_type,
                 :latest_price, :circ_mv, :amount, :turnover_rate,
                 :momentum, :volatility, :r2, :efficiency,
                 :scan_rank, :is_stale_cache, :created_at)
        """)

        date_val = pd.to_datetime(scan_date).date()
        now = datetime.now()
        rows = []
        for rank, r in enumerate(records, 1):
            rows.append({
                "scan_date": date_val,
                "symbol": r.get("symbol", ""),
                "name": r.get("name", ""),
                "asset_type": r.get("asset_type", ""),
                "latest_price": r.get("latest_price"),
                "circ_mv": r.get("circ_mv"),
                "amount": r.get("amount"),
                "turnover_rate": r.get("turnover_rate"),
                "momentum": r.get("momentum"),
                "volatility": r.get("volatility"),
                "r2": r.get("r2"),
                "efficiency": r.get("efficiency"),
                "scan_rank": rank,
                "is_stale_cache": int(is_stale_cache),
                "created_at": now,
            })

        with self.engine.begin() as conn:
            conn.execute(sql, rows)
        logger.info(f"[DB] 写入 {len(rows)} 条发现扫描结果 (scan_date={date_val})")
        return len(rows)

    def save_rebalance_log(self, record: dict) -> None:
        """
        写入一条调仓记录（INSERT OR REPLACE，幂等）。

        record 须含: trade_date, strategy, top_n, lookback_period, selected_json
        可选: trigger_type, trigger_score, signal_json
        """
        sql = text("""
            INSERT OR REPLACE INTO rebalance_log
                (trade_date, strategy, trigger_type, trigger_score,
                 top_n, lookback_period, selected_json, signal_json, created_at)
            VALUES
                (:trade_date, :strategy, :trigger_type, :trigger_score,
                 :top_n, :lookback_period, :selected_json, :signal_json, :created_at)
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {
                "trade_date": pd.to_datetime(record["trade_date"]).date(),
                "strategy": record["strategy"],
                "trigger_type": record.get("trigger_type", "daily_snapshot"),
                "trigger_score": record.get("trigger_score"),
                "top_n": record.get("top_n"),
                "lookback_period": record.get("lookback_period"),
                "selected_json": json.dumps(
                    record.get("selected", []), ensure_ascii=False, default=str
                ),
                "signal_json": json.dumps(
                    record.get("signal_details", {}), ensure_ascii=False, default=str
                ),
                "created_at": datetime.now(),
            })
        logger.info(f"[DB] 写入调仓记录 date={record['trade_date']} strategy={record['strategy']}")

    def get_latest_factors(self, top_n: int = None,
                           trade_date: str = None) -> List[dict]:
        """
        查询最新一天（或指定日期）的因子快照。

        Args:
            top_n:       若指定，只返回 is_selected=1 的前 top_n 条
            trade_date:  YYYYMMDD 或 YYYY-MM-DD；默认取库内最新日期

        Returns:
            list of dict，按 pool_rank 升序排列
        """
        with self.engine.connect() as conn:
            if trade_date is None:
                row = conn.execute(
                    text("SELECT MAX(trade_date) FROM daily_factors")
                ).fetchone()
                if not row or row[0] is None:
                    return []
                trade_date = row[0]
            else:
                trade_date = pd.to_datetime(trade_date).date()

            where = "WHERE trade_date = :d"
            if top_n is not None:
                where += " AND is_selected = 1"
            sql = text(
                f"SELECT * FROM daily_factors {where} ORDER BY pool_rank ASC"
            )
            rows = conn.execute(sql, {"d": trade_date}).mappings().fetchall()
            return [dict(r) for r in rows]

    def get_factor_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """查询某标的最近 N 天的因子历史，用于趋势分析"""
        session = self.Session()
        try:
            q = (session.query(DailyFactors)
                 .filter(DailyFactors.symbol == symbol)
                 .order_by(DailyFactors.trade_date.desc())
                 .limit(days))
            df = pd.read_sql(q.statement, session.bind)
            if not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.sort_values("trade_date", inplace=True)
            return df
        finally:
            session.close()

    def get_latest_discovery(self, scan_date: str = None) -> List[dict]:
        """
        查询最新一次（或指定日期）的发现扫描结果。

        Returns:
            list of dict，按 scan_rank 升序排列
        """
        with self.engine.connect() as conn:
            if scan_date is None:
                row = conn.execute(
                    text("SELECT MAX(scan_date) FROM discovery_scan")
                ).fetchone()
                if not row or row[0] is None:
                    return []
                scan_date = row[0]
            else:
                scan_date = pd.to_datetime(scan_date).date()

            rows = conn.execute(
                text("SELECT * FROM discovery_scan WHERE scan_date = :d ORDER BY scan_rank ASC"),
                {"d": scan_date},
            ).mappings().fetchall()
            return [dict(r) for r in rows]

    def get_rebalance_log(self, days: int = 30,
                          strategy: str = None) -> List[dict]:
        """查询最近 N 天的调仓记录"""
        cutoff = (datetime.now() - timedelta(days=days)).date()
        with self.engine.connect() as conn:
            if strategy:
                rows = conn.execute(
                    text("""SELECT * FROM rebalance_log
                            WHERE trade_date >= :c AND strategy = :s
                            ORDER BY trade_date DESC"""),
                    {"c": cutoff, "s": strategy},
                ).mappings().fetchall()
            else:
                rows = conn.execute(
                    text("""SELECT * FROM rebalance_log
                            WHERE trade_date >= :c
                            ORDER BY trade_date DESC"""),
                    {"c": cutoff},
                ).mappings().fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["selected"] = json.loads(d.get("selected_json") or "[]")
                d["signal_details"] = json.loads(d.get("signal_json") or "{}")
                result.append(d)
            return result

    # ──────────────────────────────────────────────────────────
    # 兼容旧接口
    # ──────────────────────────────────────────────────────────

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def del_tables(self):
        Base.metadata.drop_all(self.engine)

    def store_daily_data(self, symbol: str, data: pd.DataFrame):
        """兼容旧接口 → 转发到 bulk_upsert"""
        return self.bulk_upsert_daily(symbol, data)

    def get_stock_list(self) -> List[str]:
        session = self.Session()
        try:
            return [r[0] for r in session.query(StockInfo.symbol).all()]
        finally:
            session.close()

    def init_from_csv(self, csv_path: str):
        """从 CSV 初始化标的信息"""
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            session = self.Session()
            for _, row in df.iterrows():
                symbol = str(row.get("symbol", row.get("代码", ""))).zfill(6)
                name = row.get("name", row.get("名称", ""))
                if not session.query(StockInfo).filter_by(symbol=symbol).first():
                    session.add(StockInfo(
                        symbol=symbol, name=name,
                        circ_mv=row.get("circ_mv", 0),
                        industry=str(row.get("industry", "")),
                        concept=str(row.get("concept", "")),
                    ))
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"CSV import failed: {e}")


# ── Singleton ──

_db_manager = None


def get_db_manager(db_path: str = "breadfree.db") -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager
