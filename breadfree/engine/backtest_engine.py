import pandas as pd
import os
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from pyecharts.charts import Line, Grid
from pyecharts import options as opts
from ..data.data_fetcher import DataFetcher
from .broker import Broker
from ..utils.metrics import *
from ..utils.plotter import plot_backtest_results

class BacktestEngine:
    def __init__(self, strategy_cls, symbols, start_date, end_date, initial_cash=100000.0, asset_type="stock", lot_size=100, **kwargs):
        self.data_fetcher = DataFetcher(data_dir="breadfree/data/cache")
        self.broker = Broker(initial_cash=initial_cash)
        
        # 传递额外的策略参数
        self.strategy = strategy_cls(self.broker, lot_size=lot_size, **kwargs)
        
        # 支持传入单个 symbol 字符串或 symbol 列表
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols
            
        self.start_date = start_date
        self.end_date = end_date
        self.asset_type = asset_type
        self.data_map = {} # {symbol: dataframe}

    def run(self):
        # 1. Fetch Data with Warmup
        # Calculate fetch start date (45 days before start_date)
        try:
            start_dt = datetime.strptime(self.start_date, "%Y%m%d")
            fetch_start_dt = start_dt - timedelta(days=60) 
            fetch_start_date = fetch_start_dt.strftime("%Y%m%d")
        except ValueError:
            print(f"Invalid date format: {self.start_date}. Expected YYYYMMDD.")
            return

        print(f"Fetching data for {len(self.symbols)} symbols from {fetch_start_date} to {self.end_date}...")
        
        # 准备数据容器
        all_dates = set()
        warmup_data_map = {}
        backtest_data_map = {}

        for symbol in self.symbols:
            df = self.data_fetcher.fetch_data(symbol, fetch_start_date, self.end_date, asset_type=self.asset_type)
            if df.empty:
                print(f"Warning: No data for {symbol}")
                continue
                
            if not isinstance(df.index, pd.DatetimeIndex):
                 df.index = pd.to_datetime(df.index)
            
            # 存储数据
            self.data_map[symbol] = df
            
            # 分割 Warmup 和 Backtest
            warmup = df[df.index < start_dt]
            backtest = df[df.index >= start_dt]
            
            warmup_data_map[symbol] = warmup
            backtest_data_map[symbol] = backtest
            
            if not backtest.empty:
                all_dates.update(backtest.index)

        if not all_dates:
            print("No data found for backtest period.")
            return
            
        sorted_dates = sorted(list(all_dates))

        # 2. Initialize Strategy
        if hasattr(self.strategy, 'set_symbols'):
            self.strategy.set_symbols(self.symbols)
            
        # Preload history if supported
        if hasattr(self.strategy, 'preload_history'):
            # 传递 warmup 数据的字典
            self.strategy.preload_history(warmup_data_map)

        print(f"Starting backtest from {self.start_date} to {self.end_date}")
        print(f"Initial Cash: {self.broker.cash}")

        # 3. Loop through time
        last_prices = {} # 用于在某股票停牌时估算市值
        
        for date in sorted_dates:
            # 构建当天的 bars 字典
            bars = {}
            for symbol, df in backtest_data_map.items():
                if date in df.index:
                    row = df.loc[date]
                    bars[symbol] = row
                    last_prices[symbol] = row['close']
            
            if not bars:
                continue

            # Run Strategy
            # on_bar 接收 (date, bars_dict)
            self.strategy.on_bar(date, bars)
            
            # Record Equity
            # 使用 last_prices 确保即使某只股票今天无交易，也能计算持仓市值
            equity = self.broker.get_total_equity(last_prices)
            self.broker.equity_curve.append({'date': date, 'equity': equity})

        # 4. Final Results
        if not self.broker.equity_curve:
            print("No trades or equity data.")
            return

        final_equity = self.broker.equity_curve[-1]['equity']
        equity_series = pd.Series([d['equity'] for d in self.broker.equity_curve])
        
        # 使用统一的工具函数计算指标
        final_return = calculate_total_return(equity_series, initial_capital=self.broker.initial_cash)
        max_drawdown = calculate_max_drawdown(equity_series)
        sharpe_ratio = calculate_sharpe_ratio(equity_series)
        annualized_return = calculate_annualized_return(equity_series, annual_days=242)
        
        # 5. Calculate Trade Statistics
        trade_returns = [t['return_pct'] for t in self.broker.closed_trades]
        win_rate, win_count, total_trades = calculate_win_rate(trade_returns)
        # calculate_calmar_ratio
        calmar_ratio = calculate_calmar_ratio(annualized_return, abs(max_drawdown), risk_free_rate=0.015)

        print(f"Backtest finished.")
        print(f"Final Equity: {final_equity:.2f}")
        print(f"Total Return: {final_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%} ({win_count}/{total_trades})")

        # 6. Plot Results
        # self.plot_results()
        # self.plot_results_png()

    def plot_results_png(self, filename="./output/backtest_result.png"):
        if not self.broker.equity_curve:
            print("No results to plot.")
            return

        # 将 equity_curve 转换为 DataFrame
        equity_df = pd.DataFrame(self.broker.equity_curve)
        equity_df.set_index('date', inplace=True)

        # For simplicity, we use the first symbol as benchmark if available
        benchmark_series = None
        first_symbol = self.symbols[0]
        if first_symbol in self.data_map:
            df = self.data_map[first_symbol]
            # Filter for backtest period
            start_dt = datetime.strptime(self.start_date, "%Y%m%d")
            df_bt = df[df.index >= start_dt]
            if not df_bt.empty:
                benchmark_series = df_bt['close']
        
        # Determine title
        if len(self.symbols) == 1:
            title = f"Backtest: {self.symbols[0]}"
        else:
            title = f"Portfolio Backtest ({len(self.symbols)} symbols)"

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'], label='Strategy Equity')

        if benchmark_series is not None:
            # Normalize benchmark series
            benchmark_series = benchmark_series / benchmark_series.iloc[0] * equity_df['equity'].iloc[0]
            # Align time indexes
            benchmark_series = benchmark_series.reindex(equity_df.index)
            plt.plot(benchmark_series.index, benchmark_series, label=f'Benchmark ({first_symbol})', alpha=0.6, linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)

        # Ensure output directory exists
        if not os.path.exists("./output"):
            os.makedirs("./output")

        # Save figure
        plt.savefig(filename)
        print(f"结果图表已保存至 {filename}")

        # Close the plot to free memory
        plt.close()

    def plot_results_html(self, filename="./output/backtest_result.html"):
        if not self.broker.equity_curve:
            print("No results to plot.")
            return

        # For simplicity, we use the first symbol as benchmark if available
        benchmark_series = None
        first_symbol = self.symbols[0]
        if first_symbol in self.data_map:
            df = self.data_map[first_symbol]
            # Filter for backtest period
            start_dt = datetime.strptime(self.start_date, "%Y%m%d")
            df_bt = df[df.index >= start_dt]
            if not df_bt.empty:
                benchmark_series = df_bt['close']
        
        # Determine title
        if len(self.symbols) == 1:
            title = f"Backtest: {self.symbols[0]}"
        else:
            title = f"Portfolio Backtest ({len(self.symbols)} symbols)"

        print(f"Plotting results to {filename}...")
        plot_backtest_results(
            equity_curve=self.broker.equity_curve,
            transaction_history=self.broker.transaction_history,
            benchmark_series=benchmark_series,
            initial_cash=self.broker.initial_cash,
            title=title,
            filename=filename
        )

        print(f"Results saved to {filename}")
