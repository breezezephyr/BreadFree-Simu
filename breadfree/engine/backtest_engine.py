"""
回测引擎 — 高性能三层存储驱动

数据加载策略:
    1. DB 预加载: 一次性从 SQLite 加载全部标的到内存 (覆盖索引, ~50ms)
    2. 增量补数据: DB 缺失的标的走 DataFetcher 降级链获取并回写 DB
    3. numpy 迭代: 回测循环中所有数据访问都是内存操作, 零 IO
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from ..data.data_fetcher import DataFetcher
from ..data.database import get_db_manager
from .broker import Broker
from ..utils.metrics import (
    calculate_total_return, calculate_max_drawdown,
    calculate_sharpe_ratio, calculate_annualized_return,
    calculate_calmar_ratio, calculate_profit_loss_ratio,
    calculate_win_rate,
)
from ..utils.plotter import plot_backtest_results


class BacktestEngine:
    """事件驱动回测引擎: 逐日推送行情给策略, 记录权益曲线并输出绩效报告."""

    def __init__(self, strategy_cls, symbols, start_date: str, end_date: str,
                 initial_cash: float = 100000.0, asset_type: str = "stock",
                 lot_size: int = 100, data_source: str = "akshare",
                 tushare_token: str = None, **kwargs):
        self.data_fetcher = DataFetcher(
            data_dir="breadfree/data/cache",
            data_source=data_source,
            tushare_token=tushare_token,
        )
        self.broker = Broker(initial_cash=initial_cash)
        self.strategy = strategy_cls(self.broker, lot_size=lot_size, **kwargs)

        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.start_date = start_date
        self.end_date = end_date
        self.asset_type = asset_type
        self.data_map: dict = {}

    # ──────────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────────

    def run(self):
        try:
            start_dt = datetime.strptime(self.start_date, "%Y%m%d")
            fetch_start = (start_dt - timedelta(days=60)).strftime("%Y%m%d")
        except ValueError:
            print(f"日期格式错误: {self.start_date}. 需要 YYYYMMDD.")
            return

        print(f"Fetching data from database for {len(self.symbols)} symbols "
              f"from {fetch_start} to {self.end_date}...")

        t_load = time.time()
        all_dates: set = set()
        warmup_map: dict = {}
        backtest_map: dict = {}
        db = get_db_manager()

        # Step 1: 尝试从 DB 批量预加载
        db.preload_symbols(self.symbols, fetch_start, self.end_date)

        for symbol in self.symbols:
            # 优先用 DB 预加载的数据
            df = db.get_preloaded(symbol)
            if df is None or df.empty:
                df = db.get_daily_data(symbol, fetch_start, self.end_date)

            if df.empty:
                print(f"Warning: No data for {symbol} in database. "
                      f"Please ensure it is imported.")
                df = self.data_fetcher.fetch_a_stock_daily(
                    symbol, fetch_start, self.end_date)
                if df.empty:
                    print(f"Error: Unable to fetch data for {symbol}. Skipping.")
                    continue
                print(f"Data for {symbol} fetched from data source, "
                      f"{len(df)} records.")

            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            self.data_map[symbol] = df
            warmup_map[symbol] = df[df.index < start_dt]
            bt = df[df.index >= start_dt]
            backtest_map[symbol] = bt
            if not bt.empty:
                all_dates.update(bt.index)

        if not all_dates:
            print("No data found for backtest period.")
            return

        sorted_dates = sorted(all_dates)

        # 初始化策略
        if hasattr(self.strategy, "set_symbols"):
            self.strategy.set_symbols(self.symbols)
        if hasattr(self.strategy, "preload_history"):
            self.strategy.preload_history(warmup_map)

        print(f"Starting backtest from {self.start_date} to {self.end_date}")
        print(f"Initial Cash: {self.broker.cash}")

        last_prices: dict = {}
        for date in sorted_dates:
            bars = {}
            for symbol, df in backtest_map.items():
                if date in df.index:
                    bars[symbol] = df.loc[date]
                    last_prices[symbol] = df.loc[date]["close"]
            if not bars:
                continue

            self.strategy.on_bar(date, bars)
            equity = self.broker.get_total_equity(last_prices)
            self.broker.equity_curve.append({"date": date, "equity": equity})

        # 输出绩效
        self._print_performance()

    # ──────────────────────────────────────────────────────────────
    # 绩效报告
    # ──────────────────────────────────────────────────────────────

    def _print_performance(self):
        if not self.broker.equity_curve:
            print("No trades or equity data.")
            return

        equity_series = pd.Series([d["equity"] for d in self.broker.equity_curve])
        trade_returns = [t["return_pct"] for t in self.broker.closed_trades]

        total_ret = calculate_total_return(equity_series, initial_capital=self.broker.initial_cash)
        annual_ret = calculate_annualized_return(equity_series, annual_days=242)
        max_dd = calculate_max_drawdown(equity_series)
        sharpe = calculate_sharpe_ratio(equity_series)
        calmar = calculate_calmar_ratio(annual_ret, max_dd, risk_free_rate=0.015)
        win_rate, wins, total = calculate_win_rate(trade_returns)
        pl_ratio = calculate_profit_loss_ratio(trade_returns)

        final_eq = self.broker.equity_curve[-1]["equity"]
        print("Backtest finished.")
        print(f"Final Equity: {final_eq:.2f}")
        print(f"Total Return: {total_ret:.2%}")
        print(f"Annualized Return: {annual_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Calmar Ratio: {calmar:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Win Rate: {win_rate:.2%} ({wins}/{total})")
        print(f"Profit/Loss Ratio: {pl_ratio:.2f}")

    # ──────────────────────────────────────────────────────────────
    # 图表输出
    # ──────────────────────────────────────────────────────────────

    def plot_results_png(self, filename: str = "./output/backtest_result.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print(f"matplotlib 不可用, 跳过绘图: {e}")
            return

        if not self.broker.equity_curve:
            print("No results to plot.")
            return

        equity_df = pd.DataFrame(self.broker.equity_curve).set_index("date")
        benchmark_series = self._get_benchmark_series()

        title = (f"Backtest: {self.symbols[0]}" if len(self.symbols) == 1
                 else f"Portfolio Backtest ({len(self.symbols)} symbols)")

        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df["equity"], label="Strategy Equity")
        if benchmark_series is not None:
            bm = benchmark_series / benchmark_series.iloc[0] * equity_df["equity"].iloc[0]
            bm = bm.reindex(equity_df.index)
            plt.plot(bm.index, bm, label=f"Benchmark ({self.symbols[0]})",
                     alpha=0.6, linestyle="--")

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(filename) or "./output", exist_ok=True)
        plt.savefig(filename)
        plt.close()
        print(f"Result chart saved to {filename}")

    def plot_results_html(self, filename: str = "./output/backtest_result.html"):
        if not self.broker.equity_curve:
            print("No results to plot.")
            return

        benchmark_series = self._get_benchmark_series()
        title = (f"Backtest: {self.symbols[0]}" if len(self.symbols) == 1
                 else f"Portfolio Backtest ({len(self.symbols)} symbols)")

        print(f"Plotting results to {filename}...")
        plot_backtest_results(
            equity_curve=self.broker.equity_curve,
            transaction_history=self.broker.transaction_history,
            benchmark_series=benchmark_series,
            initial_cash=self.broker.initial_cash,
            title=title,
            filename=filename,
        )
        print(f"Results saved to {filename}")

    def _get_benchmark_series(self):
        """取第一只标的的收盘价序列作为基准"""
        first = self.symbols[0]
        if first not in self.data_map:
            return None
        df = self.data_map[first]
        start_dt = datetime.strptime(self.start_date, "%Y%m%d")
        bt = df[df.index >= start_dt]
        return bt["close"] if not bt.empty else None
