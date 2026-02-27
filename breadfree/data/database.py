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
