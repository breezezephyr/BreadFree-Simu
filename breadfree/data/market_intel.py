"""
MarketIntel — 多维市场情报聚合模块

提供三层情报:
  L1 资金面: 大盘主力资金流向、ETF 个股资金流向
  L2 外资面: 沪深港通北向资金流向
  L3 行业面: 行业资金流排名
  L4 市场宽度: 从 ETF 池价格数据计算的宽度指标

所有数据均来自 AkShare / 东方财富，支持历史回溯（回测安全）。
"""

import os
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "intel")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _load_etf_names() -> Dict[str, str]:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("etf_pool", {})
    except Exception:
        return {}


_ETF_NAMES: Dict[str, str] = _load_etf_names()


def _n(symbol: str) -> str:
    name = _ETF_NAMES.get(symbol, "")
    return f"{symbol}-{name}" if name else symbol


class MarketIntel:
    """市场情报聚合器"""

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.cache_dir = cache_dir
        self._market_flow_cache: Optional[pd.DataFrame] = None
        self._north_flow_cache: Optional[pd.DataFrame] = None
        self._etf_flow_cache: Dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────
    # L1: 大盘资金流向 (主力/超大单/大单/中单/小单)
    # ──────────────────────────────────────────────

    def fetch_market_fund_flow(self) -> pd.DataFrame:
        """获取大盘整体资金流向历史数据"""
        if self._market_flow_cache is not None:
            return self._market_flow_cache

        cache_file = os.path.join(self.cache_dir, "market_fund_flow.csv")
        if os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 12:
                df = pd.read_csv(cache_file, parse_dates=["日期"])
                self._market_flow_cache = df
                return df

        try:
            import akshare as ak
            df = ak.stock_market_fund_flow()
            df["日期"] = pd.to_datetime(df["日期"])
            df.to_csv(cache_file, index=False)
            self._market_flow_cache = df
            logger.info(f"[MarketIntel] Fetched market fund flow: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"[MarketIntel] market fund flow error: {e}")
            return pd.DataFrame()

    def get_market_flow_signal(self, date: pd.Timestamp, lookback: int = 5) -> Dict:
        """获取指定日期的大盘资金面信号"""
        df = self.fetch_market_fund_flow()
        if df.empty:
            return {"available": False}

        df_sorted = df.sort_values("日期")
        mask = df_sorted["日期"] <= pd.Timestamp(date)
        recent = df_sorted[mask].tail(lookback)
        if recent.empty:
            return {"available": False}

        latest = recent.iloc[-1]
        main_flow = float(latest.get("主力净流入-净额", 0))
        main_pct = float(latest.get("主力净流入-净占比", 0))

        recent_flows = recent["主力净流入-净额"].astype(float)
        avg_flow = float(recent_flows.mean())
        consecutive_inflow = int((recent_flows > 0).sum())

        return {
            "available": True,
            "main_net_flow_yi": round(main_flow / 1e8, 2),
            "main_net_pct": round(main_pct, 2),
            "avg_5d_flow_yi": round(avg_flow / 1e8, 2),
            "consecutive_inflow_days": consecutive_inflow,
            "flow_trend": "inflow" if consecutive_inflow >= 3 else ("outflow" if consecutive_inflow <= 1 else "mixed"),
        }

    # ──────────────────────────────────────────────
    # L2: ETF 个股资金流向
    # ──────────────────────────────────────────────

    def fetch_etf_fund_flow(self, symbol: str) -> pd.DataFrame:
        """获取单只 ETF 的资金流向"""
        if symbol in self._etf_flow_cache:
            return self._etf_flow_cache[symbol]

        cache_file = os.path.join(self.cache_dir, f"etf_flow_{symbol}.csv")
        if os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 12:
                df = pd.read_csv(cache_file, parse_dates=["日期"])
                self._etf_flow_cache[symbol] = df
                return df

        try:
            import akshare as ak
            market = "sh" if symbol.startswith(("5", "6")) else "sz"
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            df["日期"] = pd.to_datetime(df["日期"])
            df.to_csv(cache_file, index=False)
            self._etf_flow_cache[symbol] = df
            logger.info(f"[MarketIntel] Fetched ETF flow for {symbol}: {len(df)} rows")
            time.sleep(0.5)
            return df
        except Exception as e:
            logger.warning(f"[MarketIntel] ETF flow error for {symbol}: {e}")
            return pd.DataFrame()

    def get_etf_flow_signal(self, symbol: str, date: pd.Timestamp, lookback: int = 5) -> Dict:
        """获取单只 ETF 的资金面信号"""
        df = self.fetch_etf_fund_flow(symbol)
        if df.empty:
            return {"available": False}

        df_sorted = df.sort_values("日期")
        mask = df_sorted["日期"] <= pd.Timestamp(date)
        recent = df_sorted[mask].tail(lookback)
        if recent.empty:
            return {"available": False}

        latest = recent.iloc[-1]
        main_flow = float(latest.get("主力净流入-净额", 0))
        main_pct = float(latest.get("主力净流入-净占比", 0))

        recent_flows = recent["主力净流入-净额"].astype(float)
        avg_flow = float(recent_flows.mean())
        consecutive_inflow = int((recent_flows > 0).sum())

        return {
            "available": True,
            "main_net_flow_wan": round(main_flow / 1e4, 1),
            "main_net_pct": round(main_pct, 2),
            "avg_5d_flow_wan": round(avg_flow / 1e4, 1),
            "consecutive_inflow": consecutive_inflow,
            "fund_momentum": "strong_inflow" if consecutive_inflow >= 4 else
                             ("inflow" if consecutive_inflow >= 3 else
                              ("outflow" if consecutive_inflow <= 1 else "neutral")),
        }

    # ──────────────────────────────────────────────
    # L3: 北向资金
    # ──────────────────────────────────────────────

    def fetch_north_flow(self) -> pd.DataFrame:
        """获取沪股通历史数据"""
        if self._north_flow_cache is not None:
            return self._north_flow_cache

        cache_file = os.path.join(self.cache_dir, "north_flow.csv")
        if os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 12:
                df = pd.read_csv(cache_file, parse_dates=["日期"])
                self._north_flow_cache = df
                return df

        try:
            import akshare as ak
            df = ak.stock_hsgt_hist_em(symbol="沪股通")
            df["日期"] = pd.to_datetime(df["日期"])
            df.to_csv(cache_file, index=False)
            self._north_flow_cache = df
            logger.info(f"[MarketIntel] Fetched north flow: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"[MarketIntel] north flow error: {e}")
            return pd.DataFrame()

    def get_north_flow_signal(self, date: pd.Timestamp, lookback: int = 5) -> Dict:
        """北向资金信号"""
        df = self.fetch_north_flow()
        if df.empty:
            return {"available": False}

        df_sorted = df.sort_values("日期")
        mask = df_sorted["日期"] <= pd.Timestamp(date)
        recent = df_sorted[mask].tail(lookback)
        if recent.empty or "当日成交净买额" not in recent.columns:
            return {"available": False}

        flows = recent["当日成交净买额"].astype(float)
        valid_flows = flows.dropna()
        if valid_flows.empty:
            return {"available": False}

        latest_flow = float(valid_flows.iloc[-1])
        avg_flow = float(valid_flows.mean())
        consecutive_buy = int((valid_flows > 0).sum())

        return {
            "available": True,
            "latest_net_buy_yi": round(latest_flow / 1e8, 2) if abs(latest_flow) > 1e6 else round(latest_flow, 2),
            "avg_5d_yi": round(avg_flow / 1e8, 2) if abs(avg_flow) > 1e6 else round(avg_flow, 2),
            "consecutive_buy_days": consecutive_buy,
            "north_sentiment": "strong_buy" if consecutive_buy >= 4 else
                               ("buying" if consecutive_buy >= 3 else
                                ("selling" if consecutive_buy <= 1 else "neutral")),
        }

    # ──────────────────────────────────────────────
    # L4: 市场宽度（从已有价格数据计算）
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_breadth(all_metrics: Dict[str, Dict], lookback: int = 20) -> Dict:
        """从 ETF 池指标计算市场宽度"""
        if not all_metrics:
            return {}

        momentums = [m.get("momentum_20d", m.get("momentum", 0)) for m in all_metrics.values()]
        efficiencies = [m["efficiency"] for m in all_metrics.values()]
        accels = [m.get("momentum_accel", 0) for m in all_metrics.values()]
        vols = [m.get("volatility", 0) for m in all_metrics.values()]
        n = len(momentums)

        pct_up = sum(1 for m in momentums if m > 0.005) / n
        pct_strong = sum(1 for e in efficiencies if e > 1.0) / n
        pct_accel = sum(1 for a in accels if a > 0) / n
        avg_eff = float(np.mean(efficiencies))
        max_eff = float(max(efficiencies))
        eff_dispersion = float(np.std(efficiencies))
        avg_vol = float(np.mean(vols))

        concentration = max_eff / (avg_eff + 1e-6) if avg_eff > 0 else 0

        return {
            "pct_rising": round(pct_up, 2),
            "pct_strong_trend": round(pct_strong, 2),
            "pct_accelerating": round(pct_accel, 2),
            "avg_efficiency": round(avg_eff, 2),
            "max_efficiency": round(max_eff, 2),
            "efficiency_dispersion": round(eff_dispersion, 2),
            "concentration_ratio": round(concentration, 2),
            "avg_volatility": round(avg_vol, 4),
        }

    # ──────────────────────────────────────────────
    # 汇总：生成完整情报摘要（供 LLM 使用）
    # ──────────────────────────────────────────────

    def generate_intel_summary(self, date: pd.Timestamp, all_metrics: Dict[str, Dict],
                                etf_symbols: List[str] = None) -> str:
        """生成结构化市场情报摘要文本"""
        lines = []

        # 大盘资金面
        mf = self.get_market_flow_signal(date)
        if mf.get("available"):
            lines.append(f"【大盘资金面】主力净流入{mf['main_net_flow_yi']}亿(占比{mf['main_net_pct']}%), "
                         f"近5日均值{mf['avg_5d_flow_yi']}亿, "
                         f"连续流入{mf['consecutive_inflow_days']}天 → {mf['flow_trend']}")

        # 北向资金
        nf = self.get_north_flow_signal(date)
        if nf.get("available"):
            lines.append(f"【北向资金】净买入{nf['latest_net_buy_yi']}亿, "
                         f"近5日均值{nf['avg_5d_yi']}亿, "
                         f"连续买入{nf['consecutive_buy_days']}天 → {nf['north_sentiment']}")

        # ETF 个股资金流
        if etf_symbols:
            etf_flow_lines = []
            for sym in etf_symbols[:5]:
                ef = self.get_etf_flow_signal(sym, date)
                if ef.get("available"):
                    etf_flow_lines.append(
                        f"  {_n(sym)}: 主力{ef['main_net_flow_wan']:+.0f}万({ef['main_net_pct']:+.1f}%), "
                        f"近5日{ef['avg_5d_flow_wan']:+.0f}万, {ef['fund_momentum']}")
            if etf_flow_lines:
                lines.append("【ETF资金流向】\n" + "\n".join(etf_flow_lines))

        # 市场宽度
        breadth = self.compute_breadth(all_metrics)
        if breadth:
            lines.append(
                f"【市场宽度】上涨占比{breadth['pct_rising']:.0%}, "
                f"强趋势占比{breadth['pct_strong_trend']:.0%}, "
                f"加速占比{breadth['pct_accelerating']:.0%}, "
                f"平均效率{breadth['avg_efficiency']:.2f}, "
                f"离散度{breadth['efficiency_dispersion']:.2f}, "
                f"集中度{breadth['concentration_ratio']:.1f}")

        if not lines:
            return "暂无市场情报数据"
        return "\n".join(lines)

    def get_regime_enhanced(self, date: pd.Timestamp, all_metrics: Dict[str, Dict]) -> str:
        """增强版市场政权分类：结合资金面 + 价格面"""
        breadth = self.compute_breadth(all_metrics)
        mf = self.get_market_flow_signal(date)
        nf = self.get_north_flow_signal(date)

        score = 0.0

        # 价格面得分 (0-5)
        pct_up = breadth.get("pct_rising", 0.5)
        avg_eff = breadth.get("avg_efficiency", 0)
        score += pct_up * 3
        score += min(avg_eff, 1.0) * 2

        # 资金面得分 (0-3)
        if mf.get("available"):
            flow_trend = mf.get("flow_trend", "mixed")
            if flow_trend == "inflow":
                score += 2
            elif flow_trend == "mixed":
                score += 1

        # 北向资金得分 (0-2)
        if nf.get("available"):
            north = nf.get("north_sentiment", "neutral")
            if north in ("strong_buy", "buying"):
                score += 2
            elif north == "neutral":
                score += 1

        if score >= 8:
            return "strong_bull"
        elif score >= 6:
            return "bull"
        elif score >= 4:
            return "selective_bull"
        elif score >= 2.5:
            return "choppy"
        else:
            return "bear"
