"""
数据库模型 — SQLAlchemy ORM

表结构:
    stock_info           标的基本信息
    daily_data           日线 OHLCV (核心)
    monthly_data         月线 OHLCV
    technical_indicators 技术指标
    news_articles        新闻文章
    market_intel_daily   市场情报 (资金流向/北向)
    sector_info          板块信息
    stock_sector_mapping 标的-板块映射
    sentiment_data       情感分析 (预留)

索引策略 (参考高频量化):
    daily_data: 覆盖索引 (symbol, trade_date) 含 OHLCV → 回测扫描零随机IO
    news_articles: (symbol, publish_date) → 按日期范围查新闻
    market_intel_daily: (intel_type, date) → 按类型+日期查情报
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Text, Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ═══════════════════════════════════════════════════════════════
# 核心行情表
# ═══════════════════════════════════════════════════════════════

class StockInfo(Base):
    """标的基本信息"""
    __tablename__ = "stock_info"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(50), nullable=False)
    market = Column(String(10))
    industry = Column(String(50))
    concept = Column(String(200))
    circ_mv = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DailyData(Base):
    """日线 OHLCV — 回测核心数据源"""
    __tablename__ = "daily_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    amplitude = Column(Float)
    pct_chg = Column(Float)
    change = Column(Float)
    turnover = Column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_symbol_date"),
        # 覆盖索引: 回测只需 symbol+date+OHLCV, 全部在索引里 → 零表回查
        Index("ix_daily_cover", "symbol", "trade_date", "open", "close", "high", "low", "volume"),
    )


class MonthlyData(Base):
    """月线数据"""
    __tablename__ = "monthly_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    pct_chg = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_monthly_symbol_date"),
    )


class TechnicalIndicators(Base):
    """技术指标"""
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    trade_date = Column(Date, nullable=False)
    ema_5 = Column(Float)
    ema_10 = Column(Float)
    ema_20 = Column(Float)
    ema_30 = Column(Float)
    ema_60 = Column(Float)
    ema_120 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    rsi = Column(Float)
    kdj_k = Column(Float)
    kdj_d = Column(Float)
    kdj_j = Column(Float)
    boll_upper = Column(Float)
    boll_middle = Column(Float)
    boll_lower = Column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_tech_symbol_date"),
    )


# ═══════════════════════════════════════════════════════════════
# 新闻 & 情报表
# ═══════════════════════════════════════════════════════════════

class NewsArticle(Base):
    """新闻文章持久化"""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    publish_date = Column(DateTime, nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(500))
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_news_sym_date", "symbol", "publish_date"),
        UniqueConstraint("symbol", "title", "publish_date", name="uq_news_sym_title_date"),
    )


class MarketIntelDaily(Base):
    """市场情报日线 (资金流向/北向资金等)"""
    __tablename__ = "market_intel_daily"

    id = Column(Integer, primary_key=True)
    intel_type = Column(String(30), nullable=False)  # market_flow / north_flow / etf_flow
    symbol = Column(String(10))  # NULL for market-level, symbol for ETF-level
    date = Column(Date, nullable=False)
    data_json = Column(Text, nullable=False)  # 完整数据序列化为 JSON
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_intel_type_date", "intel_type", "date"),
        UniqueConstraint("intel_type", "symbol", "date", name="uq_intel_type_sym_date"),
    )


# ═══════════════════════════════════════════════════════════════
# 板块 & 情感 (保留)
# ═══════════════════════════════════════════════════════════════

class SectorInfo(Base):
    __tablename__ = "sector_info"
    id = Column(Integer, primary_key=True)
    sector_type = Column(String(20), nullable=False)
    sector_name = Column(String(50), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    __table_args__ = (
        UniqueConstraint("sector_type", "sector_name", name="uq_sector_type_name"),
    )


class StockSectorMapping(Base):
    __tablename__ = "stock_sector_mapping"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    sector_type = Column(String(20), nullable=False)
    sector_name = Column(String(50), nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "sector_type", "sector_name", name="uq_symbol_sector"),
    )


class SentimentData(Base):
    __tablename__ = "sentiment_data"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    news_date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Float)
    news_count = Column(Integer)
    keywords = Column(Text)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
