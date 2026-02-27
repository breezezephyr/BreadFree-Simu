"""
DataFetcher — 多源动态降级数据获取

降级链 (每层失败自动切换下一层):
    L1: curl_cffi 东方财富 K 线   (最快, 但 Cloud VM 有时被封)
    L2: 腾讯证券 K 线 API         (稳定, 支持前复权+日期范围)
    L3: 新浪财经 K 线 API         (稳定, 仅支持最近 N 条)
    L4: AkShare                   (全功能, 但 Cloud VM 常断连)

每层独立 try-catch, 一层成功即返回, 全部失败返回空 DataFrame.
成功获取的数据自动写入 CSV 缓存, 下次直接读缓存.
"""

import json
import os
import random
import time
import urllib.request
from typing import Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════
# L1: 东方财富 curl_cffi
# ═══════════════════════════════════════════════════════════════

_EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_KLINE_PARAMS = {
    "fields1": "f1,f2,f3,f4,f5,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
    "ut": "7eea3edcaed734bea9cbfc24409ed989",
    "klt": "101",
    "fqt": "1",
}


def _market_id(symbol: str) -> int:
    """沪市=1, 深市=0"""
    return 1 if symbol.startswith(("5", "6")) else 0


def _fetch_eastmoney(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """L1: 东方财富 curl_cffi 直连"""
    try:
        from curl_cffi import requests as cffi_req
    except ImportError:
        return pd.DataFrame()

    fqt = {"qfq": "1", "hfq": "2"}.get(adjust, "0")
    params = {**_EM_KLINE_PARAMS, "fqt": fqt, "beg": start, "end": end,
              "secid": f"{_market_id(symbol)}.{symbol}"}
    try:
        r = cffi_req.get(_EM_KLINE_URL, params=params, timeout=15, impersonate="chrome")
        data = r.json()
    except Exception:
        return pd.DataFrame()

    klines = (data.get("data") or {}).get("klines")
    if not klines:
        return pd.DataFrame()
    return _parse_em_klines(klines)


def _parse_em_klines(klines: list) -> pd.DataFrame:
    rows = [item.split(",") for item in klines]
    cols = ["date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "pct_chg", "change", "turnover"]
    if rows and len(rows[0]) > len(cols):
        cols += [f"extra_{i}" for i in range(len(rows[0]) - len(cols))]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════
# L2: 腾讯证券 (支持日期范围+前复权)
# ═══════════════════════════════════════════════════════════════

def _fetch_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """L2: 腾讯证券前复权日 K 线"""
    mkt = "sh" if symbol.startswith(("5", "6")) else "sz"
    s_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    e_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={mkt}{symbol},day,{s_fmt},{e_fmt},500,qfq")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
    except Exception:
        return pd.DataFrame()

    inner = (data.get("data") or {}).get(f"{mkt}{symbol}", {})
    kdata = inner.get("qfqday", inner.get("day", []))
    if not kdata:
        return pd.DataFrame()

    rows = []
    for bar in kdata:
        if len(bar) >= 6:
            rows.append({
                "date": bar[0], "open": bar[1], "close": bar[2],
                "high": bar[3], "low": bar[4], "volume": bar[5],
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════
# L3: 新浪财经 (仅支持最近 N 条)
# ═══════════════════════════════════════════════════════════════

def _fetch_sina(symbol: str, start: str, end: str, datalen: int = 800) -> pd.DataFrame:
    """L3: 新浪财经日 K 线 (最近 datalen 条, 无前复权)"""
    mkt = "sh" if symbol.startswith(("5", "6")) else "sz"
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={mkt}{symbol}&scale=240&ma=no&datalen={datalen}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.rename(columns={"day": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    for c in ["open", "close", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 按日期范围过滤
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    df = df[(df.index >= s) & (df.index <= e)]
    return df


# ═══════════════════════════════════════════════════════════════
# L4: AkShare (全功能兜底)
# ═══════════════════════════════════════════════════════════════

def _fetch_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """L4: AkShare (带重试)"""
    import akshare as ak
    is_etf = symbol.startswith(("5", "1"))
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = 8 + random.uniform(0, 3)
            logger.info(f"AkShare retry {attempt}/{max_retries} after {wait:.0f}s...")
            time.sleep(wait)
        try:
            if is_etf:
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                         start_date=start, end_date=end, adjust="qfq")
            else:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                        start_date=start, end_date=end, adjust="qfq")
            if df.empty:
                return pd.DataFrame()
            df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
                "涨跌额": "change", "换手率": "turnover",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.warning(f"AkShare error (attempt {attempt + 1}): {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# DataFetcher — 统一入口 + 降级链
# ═══════════════════════════════════════════════════════════════

# 降级链顺序 (每层名称 + 函数)
_FETCH_CHAIN = [
    ("eastmoney", _fetch_eastmoney),
    ("tencent", _fetch_tencent),
    ("sina", _fetch_sina),
    ("akshare", _fetch_akshare),
]


class DataFetcher:
    """多源动态降级数据获取器"""

    def __init__(self, data_dir: str = "data_cache", data_source: str = "akshare",
                 tushare_token: str = None):
        self.data_dir = data_dir
        self.data_source = data_source
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        os.makedirs(self.data_dir, exist_ok=True)

        # 统计各源成功/失败次数 (运行期间)
        self._source_stats = {name: {"ok": 0, "fail": 0} for name, _ in _FETCH_CHAIN}

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   asset_type: str = "stock") -> pd.DataFrame:
        if asset_type == "stock":
            return self.fetch_a_stock_daily(symbol, start_date, end_date)
        elif asset_type == "gold":
            return self._fetch_gold(symbol, start_date, end_date)
        else:
            logger.warning(f"Unknown asset type: {asset_type}")
            return pd.DataFrame()

    def fetch_a_stock_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取 A 股/ETF 日线数据 — 先查缓存, 再走降级链"""
        cache_file = os.path.join(self.data_dir, f"{symbol}_{start_date}_{end_date}.csv")

        if os.path.exists(cache_file):
            logger.debug(f"Cache hit: {cache_file}")
            print(f"Loading data from cache: {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # 降级链: 逐层尝试
        df = pd.DataFrame()
        for source_name, fetch_fn in _FETCH_CHAIN:
            try:
                time.sleep(0.5 + random.uniform(0, 1))  # 礼貌性延迟
                df = fetch_fn(symbol, start_date, end_date)
                if not df.empty and len(df) >= 5:
                    self._source_stats[source_name]["ok"] += 1
                    print(f"Fetched data for {symbol} via {source_name}.")
                    break
                else:
                    self._source_stats[source_name]["fail"] += 1
                    df = pd.DataFrame()
            except Exception as e:
                self._source_stats[source_name]["fail"] += 1
                logger.warning(f"[{source_name}] {symbol} failed: {e}")

        if not df.empty:
            # 确保必需列存在
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = 0
            df.to_csv(cache_file)
            print(f"Data for {symbol} fetched from data source, {len(df)} records.")
        else:
            logger.error(f"All sources failed for {symbol}")
            print(f"Error: Unable to fetch data for {symbol}. Skipping.")

        return df

    def get_source_stats(self) -> dict:
        """获取各数据源成功/失败统计"""
        return dict(self._source_stats)

    def _fetch_gold(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """上海金交所现货数据 (仅 AkShare 支持)"""
        cache_file = os.path.join(self.data_dir, f"gold_{symbol}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        try:
            import akshare as ak
            df = ak.spot_hist_sge(symbol=symbol)
            if df.empty:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            s = pd.to_datetime(start_date)
            e = pd.to_datetime(end_date)
            df = df[(df.index >= s) & (df.index <= e)]
            if "volume" not in df.columns:
                df["volume"] = 0
            if not df.empty:
                df.to_csv(cache_file)
            return df
        except Exception as e:
            logger.error(f"Gold data error: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = DataFetcher(data_dir=os.path.join(os.path.dirname(__file__), "cache"))
    df = fetcher.fetch_a_stock_daily("510300", "20250101", "20251231")
    print(f"Rows: {len(df)}")
    print(df.head() if not df.empty else "No data")
    print(f"\nSource stats: {fetcher.get_source_stats()}")
