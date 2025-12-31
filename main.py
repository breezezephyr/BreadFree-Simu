import sys
import os
from datetime import datetime, timedelta

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.ma_strategy import DoubleMAStrategy
from breadfree.strategies.agent_strategy import AgentStrategy
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.data.stock_pool import STOCK_POOLS

def main():
    # Configuration for Stock
    # 建议从 breadfree/data/stock_pool.py 中选择
    # 1. 宽基 ETF (推荐): "510050" (上证50), "588000" (科创50)
    # 2. 科技成长: "300750" (宁德时代), "688981" (中芯国际)
    # 3. 稳健白马: "600519" (茅台), "600036" (招行)
    
    symbol = STOCK_POOLS["Growth_Tech"][0] 
    
    # 回测最近一周，以匹配刚抓取的新闻数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    initial_cash = 1000000.0
    asset_type = "stock"
    lot_size = 100

    # Configuration for Gold (Au99.99)
    # symbol = "Au99.99"
    # # Shorten the period for LLM backtest to save tokens and time
    # start_date = "20231201" 
    # end_date = "20231205"
    # initial_cash = 500000.0 
    # asset_type = "gold"
    # lot_size = 100 

    # Choose Strategy
    # strategy_cls = DoubleMAStrategy
    # strategy_cls = BenchmarkStrategy # Market Benchmark (Buy & Hold)
    strategy_cls = AgentStrategy

    print(f"Running backtest with {strategy_cls.__name__}...")

    # Create and run backtest
    engine = BacktestEngine(
        strategy_cls=strategy_cls,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        asset_type=asset_type,
        lot_size=lot_size
    )
    
    engine.run()
    
    # Plot results (generates HTML file)
    try:
        engine.plot_results("backtest_result.html")
    except Exception as e:
        print(f"Could not plot results: {e}")

if __name__ == "__main__":
    main()
