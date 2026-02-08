import pandas as pd
import time
import os
import random

# 东方财富 A 股实时行情接口（与 akshare 一致），用 curl_cffi 可避免 RemoteDisconnected
_SPOT_EM_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
_SPOT_EM_PARAMS = {
    "pn": "1",
    "pz": "100",
    "po": "1",
    "np": "1",
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": "2",
    "invt": "2",
    "fid": "f12",
    "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
    "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
    "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
}
_SPOT_COLUMNS = [
    "序号", "代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅",
    "最高", "最低", "今开", "昨收", "量比", "换手率", "市盈率-动态", "市净率",
    "总市值", "流通市值", "涨速", "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅",
]


def _fetch_spot_em_curl() -> pd.DataFrame:
    """用 curl_cffi 请求东方财富 A 股实时行情，每页带重试。"""
    try:
        from curl_cffi import requests as cffi_req
    except ImportError:
        return pd.DataFrame()
    all_rows = []
    page = 1
    total_pages = 1
    time.sleep(5 + random.uniform(0, 3))
    retry_delays = (5, 15, 30)
    while page <= total_pages:
        params = {**_SPOT_EM_PARAMS, "pn": str(page)}
        data = None
        for retry in range(3):
            if retry > 0:
                w = retry_delays[min(retry - 1, 2)] + random.uniform(0, 3)
                print(f"  curl retry page {page} after {w:.1f}s...")
                time.sleep(w)
            try:
                r = cffi_req.get(_SPOT_EM_URL, params=params, timeout=25, impersonate="chrome")
                data = r.json()
                break
            except Exception as e:
                print(f"Request page {page} failed (attempt {retry + 1}/3): {e}")
        if data is None:
            break
        if not data.get("data") or "diff" not in data["data"]:
            break
        diff = data["data"]["diff"]
        if not diff:
            break
        if page == 1:
            total = data["data"].get("total", 0)
            total_pages = max(1, (total + 99) // 100)
            if total_pages > 50:
                total_pages = 50
        all_rows.extend(diff)
        if page < total_pages:
            time.sleep(1 + random.uniform(0.5, 1.5))
        page += 1
    if not all_rows:
        return pd.DataFrame()
    temp_df = pd.DataFrame(all_rows)
    # 列名映射（与 akshare stock_zh_a_spot_em 一致）
    col_map = {
        "f1": "序号", "f2": "_", "f3": "最新价", "f4": "涨跌额", "f5": "成交量", "f6": "成交额",
        "f7": "振幅", "f8": "换手率", "f9": "市盈率-动态", "f10": "量比", "f11": "5分钟涨跌",
        "f12": "代码", "f13": "_2", "f14": "名称", "f15": "最高", "f16": "最低", "f17": "今开",
        "f18": "昨收", "f20": "总市值", "f21": "流通市值", "f22": "年初至今涨跌幅",
        "f23": "涨速", "f24": "市净率", "f25": "60日涨跌幅",
        "f62": "-", "f128": "-", "f136": "-", "f115": "-", "f152": "-",
    }
    temp_df = temp_df.rename(columns=col_map)
    for col in ["最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "最高", "最低", "今开", "昨收", "总市值", "流通市值"]:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
    if "流通市值" in temp_df.columns:
        temp_df = temp_df.sort_values(by="流通市值", ascending=False).reset_index(drop=True)
    keep = [c for c in _SPOT_COLUMNS if c in temp_df.columns]
    return temp_df[keep] if keep else temp_df


def _fetch_spot_em_akshare() -> pd.DataFrame:
    """通过 akshare 获取（易被东方财富断连时失败）。"""
    import akshare as ak
    return ak.stock_zh_a_spot_em()


# 网络全部失败时使用的备用列表（约 150 只，含常见 ETF + 大盘股），保证脚本能产出 CSV
_FALLBACK_SYMBOLS = [
    "510300", "510500", "159915", "512880", "510880", "513100", "512660", "511260", "159949",  # config ETF
    "600519", "000858", "601318", "600036", "000333", "601888", "300750", "002594", "600030", "000651",
    "601012", "300059", "002415", "600276", "603259", "601166", "002475", "300760", "600900", "601398",
    "600887", "000568", "601288", "600585", "002304", "601899", "600309", "000725", "601628", "601857",
    "600436", "002352", "300124", "603501", "600000", "601328", "601818", "600016", "601988", "601998",
    "000001", "002142", "300347", "600048", "000002", "001979", "600104", "601633", "000063", "002049",
    "300274", "603288", "002027", "300015", "002230", "600570", "002241", "300496", "002271", "600763",
    "300122", "002007", "600196", "000538", "002001", "600660", "000963", "002008", "300003", "002044",
    "600346", "002410", "300014", "002385", "600886", "601012", "002714", "300274", "603160", "002475",
    "600031", "601100", "000157", "300498", "002352", "603259", "300760", "002594", "601012", "300059",
    "600276", "002415", "600309", "000858", "600519", "601318", "600036", "000333", "601888", "300750",
    "600030", "000651", "601899", "000725", "601628", "601857", "600436", "300124", "603501", "600585",
    "002304", "600887", "000568", "601288", "600900", "601398", "601166", "002475", "601012", "300059",
    "002415", "600276", "603259", "601166", "002475", "300760", "600900", "601398", "601888", "300750",
]


def _build_fallback_top150() -> pd.DataFrame:
    """用内置列表生成 top150 DataFrame（名称占位），供网络全挂时使用。"""
    n = min(150, len(_FALLBACK_SYMBOLS))
    codes = [_FALLBACK_SYMBOLS[i].zfill(6) if len(_FALLBACK_SYMBOLS[i]) < 6 else _FALLBACK_SYMBOLS[i] for i in range(n)]
    return pd.DataFrame({"symbol": codes, "name": ["—"] * n, "circ_mv": [0] * n})


def get_stock_zh_a_spot_em() -> pd.DataFrame:
    """获取 A 股实时行情，优先 curl_cffi，失败则重试 akshare。"""
    df = _fetch_spot_em_curl()
    if not df.empty:
        print("Fetched A-share spot via curl_cffi (East Money).")
        return df
    for attempt in range(3):
        if attempt > 0:
            wait = (5, 15, 30)[min(attempt - 1, 2)] + random.uniform(0, 3)
            print(f"Retry {attempt}/2 after {wait:.1f}s...")
            time.sleep(wait)
        try:
            df = _fetch_spot_em_akshare()
            if not df.empty:
                return df
        except Exception as e:
            print(f"akshare attempt {attempt + 1} failed: {e}")
    return pd.DataFrame()


# 获取实时行情；全部失败则用备用列表
stock_df = get_stock_zh_a_spot_em()
use_fallback = stock_df.empty
if use_fallback:
    print("Network fetch failed, using built-in fallback symbol list (no live 流通市值).")
    top_150 = _build_fallback_top150()
    top_150["industry"] = "N/A"
    top_150["concept"] = "N/A"
else:
    stock_df_sorted = stock_df.sort_values(by="流通市值", ascending=False)
    top_150 = stock_df_sorted.head(150)[["代码", "名称", "流通市值"]].copy()
    top_150.columns = ["symbol", "name", "circ_mv"]
    top_150["industry"] = ""
    top_150["concept"] = ""

if not use_fallback:
    import akshare as ak
    for idx, row in top_150.iterrows():
        symbol = str(row["symbol"]).strip().zfill(6) if len(str(row["symbol"]).strip()) < 6 else str(row["symbol"]).strip()
        print(f"Fetching sector info for {symbol}")
        try:
            info_df = ak.stock_individual_info_em(symbol=symbol)
            if info_df.empty or "item" not in info_df.columns or "value" not in info_df.columns:
                industry, concept = "N/A", "N/A"
            else:
                info_dict = dict(zip(info_df["item"], info_df["value"]))
                industry = info_dict.get("行业", "")
                concept = info_dict.get("概念", "")
            top_150.at[idx, "industry"] = industry
            top_150.at[idx, "concept"] = concept
            print(f"✅ {symbol} - Industry: {industry} | Concept: {concept}")
            time.sleep(0.3)
        except Exception as e:
            print(f"❌ Failed to fetch info for {symbol}: {e}")
            top_150.at[idx, "industry"] = "Error"
            top_150.at[idx, "concept"] = "Error"

if not os.path.exists("breadfree/data/cache/"):
    os.makedirs("breadfree/data/cache/")
    print("\n✅ Cache directory created")

top_150.to_csv("breadfree/data/cache/top_150_with_sectors.csv", index=False, encoding="utf-8-sig")
print("\n✅ Sector information saved")
