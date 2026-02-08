import akshare as ak
import pandas as pd
import os
import time
import random

# 东方财富 K 线接口（与 akshare 相同），用 curl_cffi 请求可避免 RemoteDisconnected
_EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_KLINE_PARAMS = {
    "fields1": "f1,f2,f3,f4,f5,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
    "ut": "7eea3edcaed734bea9cbfc24409ed989",
    "klt": "101",
    "fqt": "1",
}


def _fetch_em_kline_curl(symbol: str, start_date: str, end_date: str, is_etf: bool, adjust: str = "qfq") -> pd.DataFrame:
    """用 curl_cffi 请求东方财富 K 线，避免 requests 被断连。返回与 akshare 同结构的 DataFrame。"""
    try:
        from curl_cffi import requests as cffi_req
    except ImportError:
        return pd.DataFrame()
    fqt = "1" if adjust == "qfq" else ("2" if adjust == "hfq" else "0")
    if is_etf:
        market_id = 1 if symbol.startswith(("5", "6")) else 0
    else:
        market_id = 1 if symbol.startswith("6") else 0
    params = {
        **_EM_KLINE_PARAMS,
        "fqt": fqt,
        "beg": start_date,
        "end": end_date,
        "secid": f"{market_id}.{symbol}",
    }
    try:
        r = cffi_req.get(_EM_KLINE_URL, params=params, timeout=20, impersonate="chrome")
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if not (data.get("data") and data["data"].get("klines")):
        return pd.DataFrame()
    rows = [item.split(",") for item in data["data"]["klines"]]
    df = pd.DataFrame(
        rows,
        columns=["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"],
    )
    df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class DataFetcher:
    def __init__(self, data_dir="data_cache", data_source="akshare", tushare_token=None):
        self.data_dir = data_dir
        self.data_source = data_source
        # Prefer using the provided token, if not provided try to get from environment variable
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")

        if self.data_source == "tushare":
            try:
                import tushare as ts
                if self.tushare_token:
                    ts.set_token(self.tushare_token)
                else:
                    print("Warning: Tushare token not found. Please set it in config.yaml or via TUSHARE_TOKEN environment variable.")
                self.pro = ts.pro_api()
            except ImportError:
                print("Warning: tushare not installed, please install it using 'pip install tushare'")
            except Exception as e:
                print(f"Warning: Failed to initialize tushare: {e}")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data(self, symbol: str, start_date: str, end_date: str, asset_type: str = "stock") -> pd.DataFrame:
        """
        General data fetching interface
        :param symbol: Target code
        :param start_date: Start date 'YYYYMMDD'
        :param end_date: End date 'YYYYMMDD'
        :param asset_type: Asset type 'stock' or 'gold'
        :return: DataFrame
        """
        if asset_type == "stock":
            return self.fetch_a_stock_daily(symbol, start_date, end_date)
        elif asset_type == "gold":
            return self.fetch_sge_gold_daily(symbol, start_date, end_date)
        else:
            print(f"Unknown asset type: {asset_type}")
            return pd.DataFrame()

    def fetch_sge_gold_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Shanghai Gold Exchange spot data
        :param symbol: e.g. 'Au99.99'
        """
        # Format date as YYYYMMDD for cache filename, but akshare interface may not need date filtering or requires specific format
        # ak.spot_hist_sge returns all historical data, we need to filter ourselves
        cache_file = os.path.join(self.data_dir, f"gold_{symbol}_{start_date}_{end_date}.csv")
        
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        print(f"Fetching gold data for {symbol} from AkShare...")
        try:
            df = ak.spot_hist_sge(symbol=symbol)
            
            if df.empty:
                print("Warning: No data fetched.")
                return pd.DataFrame()

            # df columns: date, open, high, low, close
            # Ensure column names are consistent
            df.rename(columns={
                'date': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low'
            }, inplace=True)
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Filter dates
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            if df.empty:
                print("Warning: No data in date range.")
                return pd.DataFrame()

            # Gold data may not have volume, complete it to prevent strategy errors
            if 'volume' not in df.columns:
                df['volume'] = 0

            # Save cache
            df.to_csv(cache_file)
            return df
            
        except Exception as e:
            print(f"Error fetching gold data: {e}")
            return pd.DataFrame()

    def _get_tushare_code(self, symbol: str) -> str:
        if symbol.startswith('6') or symbol.startswith('5'):
            return f"{symbol}.SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"{symbol}.SZ"
        elif symbol.startswith('1'): # SZ ETF
            return f"{symbol}.SZ"
        elif symbol.startswith('4') or symbol.startswith('8'):
            return f"{symbol}.BJ"
        return symbol

    def _fetch_from_tushare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        import tushare as ts
        print(f"Fetching data for {symbol} from Tushare...")
        ts_code = self._get_tushare_code(symbol)
        
        try:
            # 判定资源类型
            asset_type = 'FD' if (symbol.startswith('5') or symbol.startswith('1')) else 'E'
            
            # 使用 pro_bar 获取复权数据
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date, asset=asset_type)
            
            if df is None or df.empty:
                print("Warning: No data fetched from Tushare.")
                return pd.DataFrame()

            # 重命名列以符合通用习惯
            df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume'
            }, inplace=True)
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching data from Tushare: {e}")
            return pd.DataFrame()

    def fetch_a_stock_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取A股日线数据或ETF数据
        :param symbol: 股票代码或ETF代码，例如 '000001' 或 '588000'
        :param start_date: 开始日期 'YYYYMMDD'
        :param end_date: 结束日期 'YYYYMMDD'
        :return: DataFrame
        """
        cache_file = os.path.join(self.data_dir, f"{symbol}_{start_date}_{end_date}.csv")
        
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        if self.data_source == "tushare":
            df = self._fetch_from_tushare(symbol, start_date, end_date)
        else:
            df = self._fetch_from_akshare(symbol, start_date, end_date)

        if not df.empty:
            df.to_csv(cache_file)
        
        return df

    def _fetch_from_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 优先用 curl_cffi 直连东方财富 K 线接口（模拟浏览器，减少 RemoteDisconnected）
        time.sleep(3 + random.uniform(0, 2))
        is_etf = symbol.startswith(("5", "1"))
        df = _fetch_em_kline_curl(symbol, start_date, end_date, is_etf=is_etf, adjust="qfq")
        if not df.empty:
            print(f"Fetched data for {symbol} via curl_cffi (East Money).")
            return df

        # 回退到 akshare，并做重试
        max_retries = 3
        retry_delays = (8, 20, 45)
        for attempt in range(max_retries):
            if attempt > 0:
                wait = retry_delays[min(attempt - 1, len(retry_delays) - 1)] + random.uniform(0, 3)
                print(f"Retry {attempt}/{max_retries - 1} after {wait:.1f}s...")
                time.sleep(wait)

            print(f"Fetching data for {symbol} from AkShare...")
            try:
                if is_etf:
                    df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                else:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                if df.empty:
                    return pd.DataFrame()
                df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                    "成交量": "volume", "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
                    "涨跌额": "change", "换手率": "turnover",
                }, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df
            except Exception as e:
                print(f"Error fetching data from AkShare: {e}")
                if attempt == max_retries - 1:
                    return pd.DataFrame()
        return pd.DataFrame()

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "cache/top_150_with_sectors.csv")
    if os.path.exists(path):
        top150_df = pd.read_csv(path, dtype={"symbol": str})
        print("Top 150:")
        print(top150_df)
    else:
        print(f"Top 150 file not found at {path}")
    now = pd.Timestamp.now().strftime("%Y%m%d")
    # 2005年01月01
    start = "20050101"
    print(f"Fetching sample stock data for 000001 from {start} to {now}...")

    # 测试代码
    # fetcher = DataFetcher(data_dir="breadfree/data/cache")
    # df = fetcher.fetch_a_stock_daily("000001", "20230101", "20231231")
    # print(df.head())

    # 批量获取并保存
    for _, row in top150_df.iterrows():
        code = row['symbol']
        name = row['name']
        print(f"Fetching data for {code} - {name}...")
        fetcher = DataFetcher(data_dir="breadfree/data/cache")
        df = fetcher.fetch_a_stock_daily(code, start, now)

        # break  # 仅测试第一个
