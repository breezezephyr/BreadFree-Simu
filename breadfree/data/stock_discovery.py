"""
StockDiscovery — 全市场主动选股引擎

从固定股票池扩展到全 A 股/ETF 市场扫描, 主动发现交易机会:
    1. 宽筛: 从东方财富/AkShare 拉取全市场实时行情 (流通市值, 成交额, 换手率)
    2. 流动性过滤: 日均成交额 > 阈值, 流通市值 > 阈值
    3. 量化评分: 对通过流动性筛选的标的计算效率分 (与 RotationStrategy 一致)
    4. 增量融合: 将新发现的高分标的与固定池合并, 输出扩展池

设计原则:
    - 低频调用: 每次调仓前扫描一次, 不逐日扫描
    - 降级容错: curl_cffi → AkShare → 备用列表
    - 缓存友好: 扫描结果缓存到 CSV + DB, 避免重复网络请求
    - 向后兼容: 固定池始终保留, 新发现标的作为补充
"""

import os
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "discovery")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# L1: 东方财富全市场行情 (curl_cffi)
# ═══════════════════════════════════════════════════════════════

_SPOT_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"

_STOCK_PARAMS = {
    "pn": "1", "pz": "200", "po": "1", "np": "1",
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": "2", "invt": "2", "fid": "f6",
    "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
    "fields": "f2,f3,f5,f6,f7,f8,f12,f14,f15,f16,f17,f18,f20,f21",
}

_ETF_PARAMS = {
    "pn": "1", "pz": "200", "po": "1", "np": "1",
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": "2", "invt": "2", "fid": "f6",
    "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
    "fields": "f2,f3,f5,f6,f7,f8,f12,f14,f15,f16,f17,f18,f20,f21",
}


def _fetch_spot_page(params: dict, page: int = 1) -> List[dict]:
    try:
        from curl_cffi import requests as cffi_req
    except ImportError:
        return []

    p = {**params, "pn": str(page)}
    try:
        r = cffi_req.get(_SPOT_URL, params=p, timeout=20, impersonate="chrome")
        data = r.json()
    except Exception:
        return []

    diff = (data.get("data") or {}).get("diff", [])
    return diff if diff else []


def _fetch_all_spots(params: dict, max_pages: int = 10) -> pd.DataFrame:
    all_rows = []
    for page in range(1, max_pages + 1):
        rows = _fetch_spot_page(params, page)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < int(params.get("pz", 200)):
            break
        time.sleep(0.3 + random.uniform(0, 0.5))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    col_map = {
        "f2": "latest_price", "f3": "pct_chg", "f5": "volume",
        "f6": "amount", "f7": "amplitude", "f8": "turnover_rate",
        "f12": "symbol", "f14": "name",
        "f15": "high", "f16": "low", "f17": "open", "f18": "prev_close",
        "f20": "total_mv", "f21": "circ_mv",
    }
    df = df.rename(columns=col_map)
    for c in ["latest_price", "pct_chg", "volume", "amount", "amplitude",
              "turnover_rate", "high", "low", "open", "prev_close",
              "total_mv", "circ_mv"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fetch_spots_akshare(asset_type: str = "stock") -> pd.DataFrame:
    try:
        import akshare as ak
        if asset_type == "etf":
            df = ak.fund_etf_spot_em()
        else:
            df = ak.stock_zh_a_spot_em()
        if df.empty:
            return pd.DataFrame()
        rename_map = {
            "代码": "symbol", "名称": "name", "最新价": "latest_price",
            "涨跌幅": "pct_chg", "成交量": "volume", "成交额": "amount",
            "振幅": "amplitude", "换手率": "turnover_rate",
            "最高": "high", "最低": "low", "今开": "open", "昨收": "prev_close",
            "总市值": "total_mv", "流通市值": "circ_mv",
        }
        df = df.rename(columns=rename_map)
        for c in ["latest_price", "amount", "circ_mv", "turnover_rate"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        logger.warning(f"[Discovery] AkShare spot fetch error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 量化筛选器
# ═══════════════════════════════════════════════════════════════

class StockDiscovery:
    """
    全市场主动选股引擎

    使用流程:
        1. discover() 扫描全市场, 返回通过筛选的标的列表
        2. get_expanded_pool() 将发现结果与固定池合并
    """

    def __init__(
        self,
        min_amount: float = 5e7,
        min_circ_mv: float = 5e9,
        min_turnover: float = 0.3,
        max_discover: int = 30,
        efficiency_threshold: float = 0.5,
        lookback_period: int = 20,
        cache_hours: float = 12.0,
        scan_etf: bool = True,
        scan_stock: bool = True,
    ):
        """
        Args:
            min_amount: 最低日成交额 (元), 默认 5000 万
            min_circ_mv: 最低流通市值 (元), 默认 50 亿
            min_turnover: 最低换手率 (%), 默认 0.3%
            max_discover: 最多发现标的数, 默认 30
            efficiency_threshold: 效率分阈值, 低于此值不入选
            lookback_period: 效率分回看窗口
            cache_hours: 扫描结果缓存时长 (小时)
            scan_etf: 是否扫描 ETF
            scan_stock: 是否扫描个股
        """
        self.min_amount = min_amount
        self.min_circ_mv = min_circ_mv
        self.min_turnover = min_turnover
        self.max_discover = max_discover
        self.efficiency_threshold = efficiency_threshold
        self.lookback_period = lookback_period
        self.cache_hours = cache_hours
        self.scan_etf = scan_etf
        self.scan_stock = scan_stock

        self._last_scan: Optional[pd.DataFrame] = None
        self._last_scan_time: Optional[datetime] = None

    def _is_cache_valid(self) -> bool:
        if self._last_scan is None or self._last_scan_time is None:
            return False
        age = (datetime.now() - self._last_scan_time).total_seconds() / 3600
        return age < self.cache_hours

    def _load_file_cache(self) -> Optional[pd.DataFrame]:
        cache_file = os.path.join(_CACHE_DIR, "latest_scan.csv")
        if not os.path.exists(cache_file):
            return None
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours > self.cache_hours:
            return None
        try:
            return pd.read_csv(cache_file)
        except Exception:
            return None

    def _save_file_cache(self, df: pd.DataFrame):
        cache_file = os.path.join(_CACHE_DIR, "latest_scan.csv")
        try:
            df.to_csv(cache_file, index=False)
        except Exception as e:
            logger.warning(f"[Discovery] Cache save failed: {e}")

    def _fetch_market_data(self) -> pd.DataFrame:
        """拉取全市场行情数据, 合并 ETF + 个股"""
        frames = []

        if self.scan_etf:
            logger.info("[Discovery] 扫描 ETF 行情...")
            df_etf = _fetch_all_spots(_ETF_PARAMS, max_pages=5)
            if df_etf.empty:
                df_etf = _fetch_spots_akshare("etf")
            if not df_etf.empty:
                df_etf["asset_type"] = "etf"
                frames.append(df_etf)

        if self.scan_stock:
            logger.info("[Discovery] 扫描 A 股行情...")
            df_stock = _fetch_all_spots(_STOCK_PARAMS, max_pages=15)
            if df_stock.empty:
                df_stock = _fetch_spots_akshare("stock")
            if not df_stock.empty:
                df_stock["asset_type"] = "stock"
                frames.append(df_stock)

        if not frames:
            logger.warning("[Discovery] 全市场行情获取失败")
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _apply_liquidity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """流动性过滤: 成交额, 流通市值, 换手率"""
        if df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        if "amount" in df.columns:
            mask &= df["amount"] >= self.min_amount
        if "circ_mv" in df.columns:
            mask &= df["circ_mv"] >= self.min_circ_mv
        if "turnover_rate" in df.columns:
            mask &= df["turnover_rate"] >= self.min_turnover
        if "latest_price" in df.columns:
            mask &= df["latest_price"] > 0

        filtered = df[mask].copy()
        logger.info(f"[Discovery] 流动性过滤: {len(df)} → {len(filtered)}")
        return filtered

    def _calc_efficiency_scores(
        self, candidates: pd.DataFrame, end_date: str
    ) -> List[dict]:
        """为候选标的计算效率分 (复用现有 DataFetcher + 效率分公式)。限制处理数量并打进度日志，避免 178 个串行请求卡住报告。"""
        from .data_fetcher import DataFetcher
        from .database import get_db_manager
        from ..utils.metrics import calculate_efficiency_metrics

        cfg = get_config().get("discovery", {})
        max_to_score = int(cfg.get("max_candidates_to_score", 60))
        if len(candidates) > max_to_score:
            candidates = candidates.head(max_to_score)
            logger.info(f"[Discovery] 候选数过多，仅对前 {max_to_score} 个计算效率分，避免长时间阻塞")

        fetcher = DataFetcher(
            data_dir=os.path.join(os.path.dirname(__file__), "cache"),
            data_source="akshare",
        )
        db = get_db_manager()

        lookback_days = self.lookback_period * 3
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_date = (end_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")

        scored = []
        symbols = candidates["symbol"].tolist()

        db.preload_symbols(symbols, start_date, end_date)

        total = len(candidates)
        for idx, (_, row) in enumerate(candidates.iterrows()):
            if (idx + 1) % 15 == 0 or idx == 0 or idx == total - 1:
                logger.info(f"[Discovery] 计算效率分 {idx + 1}/{total}...")
            symbol = str(row["symbol"]).zfill(6)

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

            closes = df["close"].dropna().tolist()
            if len(closes) < self.lookback_period:
                continue

            metrics = calculate_efficiency_metrics(closes, self.lookback_period)
            if metrics is None:
                continue

            if metrics["efficiency"] < self.efficiency_threshold:
                continue

            scored.append({
                "symbol": symbol,
                "name": row.get("name", symbol),
                "asset_type": row.get("asset_type", "stock"),
                "latest_price": float(row.get("latest_price", metrics["close"])),
                "circ_mv": float(row.get("circ_mv", 0)),
                "amount": float(row.get("amount", 0)),
                "turnover_rate": float(row.get("turnover_rate", 0)),
                "momentum": metrics["momentum"],
                "volatility": metrics["volatility"],
                "r2": metrics["r2"],
                "efficiency": metrics["efficiency"],
                "close": metrics["close"],
            })

        scored.sort(key=lambda x: x["efficiency"], reverse=True)
        return scored[:self.max_discover]

    def discover(self, end_date: str = None) -> List[dict]:
        """
        执行全市场扫描, 返回新发现的高效率标的列表

        Args:
            end_date: 截止日期 YYYYMMDD, 默认今天

        Returns:
            list of dict, 每个 dict 包含 symbol, name, efficiency, momentum 等
        """
        if self._is_cache_valid():
            return self._last_scan.to_dict("records") if not self._last_scan.empty else []

        file_cache = self._load_file_cache()
        if file_cache is not None and not file_cache.empty:
            self._last_scan = file_cache
            self._last_scan_time = datetime.now()
            logger.info(f"[Discovery] 从文件缓存加载 {len(file_cache)} 条扫描结果")
            return file_cache.to_dict("records")

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        logger.info("[Discovery] 开始全市场扫描...")

        market_data = self._fetch_market_data()
        if market_data.empty:
            logger.warning("[Discovery] 无法获取市场数据, 返回空结果")
            return []

        filtered = self._apply_liquidity_filter(market_data)
        if filtered.empty:
            logger.warning("[Discovery] 流动性过滤后无候选标的")
            return []

        cfg = get_config()
        fixed_pool = set(cfg.get("etf_pool", {}).keys())
        filtered = filtered[~filtered["symbol"].isin(fixed_pool)]
        logger.info(f"[Discovery] 排除固定池后剩余 {len(filtered)} 个候选")

        scored = self._calc_efficiency_scores(filtered, end_date)
        logger.info(f"[Discovery] 效率分筛选后 {len(scored)} 个标的")

        if scored:
            result_df = pd.DataFrame(scored)
            self._last_scan = result_df
            self._last_scan_time = datetime.now()
            self._save_file_cache(result_df)

        return scored

    def get_expanded_pool(self, end_date: str = None, max_expand: int = 10) -> Dict[str, str]:
        """
        将固定池与新发现标的合并, 返回扩展池 {symbol: name}

        Args:
            end_date: 截止日期
            max_expand: 最多新增标的数量

        Returns:
            合并后的标的池字典
        """
        cfg = get_config()
        pool = dict(cfg.get("etf_pool", {}))

        discovered = self.discover(end_date)
        added = 0
        new_symbols = []
        for item in discovered:
            if added >= max_expand:
                break
            sym = item["symbol"]
            if sym not in pool:
                pool[sym] = f"{item['name']}★"
                new_symbols.append(f"{sym}-{item['name']}")
                added += 1

        if new_symbols:
            logger.info(f"[Discovery] 扩展池新增 {added} 个标的: {new_symbols}")

        return pool

    def get_discovery_summary(self, end_date: str = None) -> dict:
        """生成发现摘要, 用于邮件报告"""
        discovered = self.discover(end_date)

        if not discovered:
            return {
                "total_scanned": 0,
                "total_discovered": 0,
                "top_discoveries": [],
                "avg_efficiency": 0,
                "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

        efficiencies = [d["efficiency"] for d in discovered]
        return {
            "total_scanned": len(discovered),
            "total_discovered": len(discovered),
            "top_discoveries": discovered[:10],
            "avg_efficiency": float(np.mean(efficiencies)),
            "max_efficiency": float(max(efficiencies)),
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
