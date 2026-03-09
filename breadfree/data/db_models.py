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

    ── 选股择时数据 ──
    daily_factors        每日全池因子快照 (效率分/动量/R²等)
    discovery_scan       全市场主动发现扫描结果
    rebalance_log        调仓/择时决策记录

索引策略 (参考高频量化):
    daily_data: 覆盖索引 (symbol, trade_date) 含 OHLCV → 回测扫描零随机IO
    news_articles: (symbol, publish_date) → 按日期范围查新闻
    market_intel_daily: (intel_type, date) → 按类型+日期查情报
    daily_factors: (trade_date, pool_rank) → 按日期+排名快速查最新选股
    discovery_scan: (scan_date, scan_rank) → 按日期+排名查发现结果
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


# ═══════════════════════════════════════════════════════════════
# 选股择时数据表
# ═══════════════════════════════════════════════════════════════

class DailyFactors(Base):
    """
    每日全池因子快照 — 选股核心数据

    每次每日报告（或独立快照脚本）运行时，为固定池内所有标的写入一行。
    可供下游脚本/cron 直接查询最新排名、因子趋势、选股信号。
    """
    __tablename__ = "daily_factors"

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String(10), nullable=False)
    name = Column(String(50))

    # 收盘价
    close = Column(Float)

    # 核心因子（与 metrics.py calculate_efficiency_metrics 一致）
    momentum = Column(Float)      # 区间 ROC（20日）
    volatility = Column(Float)    # 日波动率（日收益标准差）
    r2 = Column(Float)            # 线性趋势 R²
    efficiency = Column(Float)    # 效率分 = (momentum / period_vol) * R²

    # 扩展因子（若计算了 accel/composite 时填入，否则 NULL）
    accel = Column(Float)                # 动量加速度
    composite = Column(Float)            # 多因子综合分
    drawdown_from_high = Column(Float)   # 离近期高点回撤

    # 排名与选股标志
    pool_rank = Column(Integer)    # 在池内排名（1=效率分最高）
    is_selected = Column(Integer)  # 是否入选 Top-N（1=是，0=否）
    top_n = Column(Integer)        # 本次 top_n 配置
    lookback_period = Column(Integer)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("trade_date", "symbol", name="uq_factors_date_symbol"),
        Index("ix_factors_date_rank", "trade_date", "pool_rank"),
    )


class DiscoveryScan(Base):
    """
    全市场主动发现扫描结果 — 扩展选股数据

    每次 StockDiscovery.discover() 完成后写入。
    记录不在固定池内、但通过流动性+效率筛选的标的。
    """
    __tablename__ = "discovery_scan"

    id = Column(Integer, primary_key=True)
    scan_date = Column(Date, nullable=False)
    symbol = Column(String(10), nullable=False)
    name = Column(String(50))
    asset_type = Column(String(10))    # 'etf' or 'stock'

    # 实时行情字段（来自东方财富 API）
    latest_price = Column(Float)
    circ_mv = Column(Float)            # 流通市值（元）
    amount = Column(Float)             # 日成交额（元）
    turnover_rate = Column(Float)      # 换手率（%）

    # 效率因子
    momentum = Column(Float)
    volatility = Column(Float)
    r2 = Column(Float)
    efficiency = Column(Float)

    scan_rank = Column(Integer)        # 在本次扫描结果中的排名（1=最优）
    is_stale_cache = Column(Integer)   # 数据是否来自过期缓存（1=是）

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("scan_date", "symbol", name="uq_scan_date_symbol"),
        Index("ix_scan_date_rank", "scan_date", "scan_rank"),
    )


class RebalanceLog(Base):
    """
    调仓/择时决策记录 — 择时核心数据

    每次策略触发调仓时写入一行，记录：调仓日期、触发原因、选出的标的及权重。
    供下游消费最新持仓建议，或审计历史调仓决策。
    """
    __tablename__ = "rebalance_log"

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    strategy = Column(String(50), nullable=False)   # RotationStrategy / DynamicRotation / daily_report

    # 触发信息
    trigger_type = Column(String(50))     # periodic / trailing_stop / momentum_breakout / daily_snapshot
    trigger_score = Column(Float)         # 综合触发分（DynamicRotation 专用）

    # 配置
    top_n = Column(Integer)
    lookback_period = Column(Integer)

    # 选股结果（JSON 序列化）
    selected_json = Column(Text)   # [{symbol, name, rank, efficiency, weight}, ...]
    signal_json = Column(Text)     # 额外信号详情（DynamicRotation 专用）

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("trade_date", "strategy", name="uq_rebalance_date_strategy"),
        Index("ix_rebalance_date", "trade_date"),
    )
