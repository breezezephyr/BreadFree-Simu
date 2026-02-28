"""BreadFree 回测入口 — 统一 CLI 接口"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from breadfree.engine.backtest_engine import BacktestEngine
from breadfree.strategies.ma_strategy import DoubleMAStrategy
from breadfree.strategies.benchmark_strategy import BenchmarkStrategy
from breadfree.strategies.effi_rotation_strategy import RotationStrategy
from breadfree.strategies.effi_agent_strategy import EffiAgentRotationStrategy
from breadfree.strategies.triple_momentum_strategy import TripleMomentumStrategy
from breadfree.strategies.agent_strategy_v2 import AgentStrategyV2
from breadfree.strategies.dynamic_rotation_strategy import DynamicRotationStrategy

STRATEGY_MAP = {
    "DoubleMAStrategy": DoubleMAStrategy,
    "BenchmarkStrategy": BenchmarkStrategy,
    "AgentStrategyV2": AgentStrategyV2,
    "RotationStrategy": RotationStrategy,
    "EffiA": EffiAgentRotationStrategy,
    "TripleMomentumStrategy": TripleMomentumStrategy,
    "DynamicRotation": DynamicRotationStrategy,
}


def main():
    parser = argparse.ArgumentParser(description="BreadFree Backtest Engine")
    parser.add_argument(
        "--strategy", type=str, choices=list(STRATEGY_MAP.keys()),
        help="策略名称",
    )
    parser.add_argument("--start_date", type=str, help="起始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, help="结束日期 YYYYMMDD")
    parser.add_argument("--initial_cash", type=float, help="初始资金")

    # RotationStrategy / TripleMomentumStrategy 超参数
    parser.add_argument("--lookback_period", type=int, default=20, help="回看窗口 (天)")
    parser.add_argument("--hold_period", type=int, default=20, help="持仓周期 (天)")
    parser.add_argument("--top_n", type=int, help="持仓标的数量")
    parser.add_argument("--min_momentum", type=float, help="最小动量阈值")
    parser.add_argument("--use_efficiency", type=bool, choices=[True, False], default=True,
                        help="是否启用效率分")

    # TripleMomentumStrategy 专用参数
    parser.add_argument("--bias_n", type=int, help="乖离率均线窗口")
    parser.add_argument("--momentum_day", type=int, help="动量回归窗口")
    parser.add_argument("--slope_n", type=int, help="斜率/效率窗口")
    parser.add_argument("--rebalance_threshold", type=float, help="调仓阈值倍数")

    # DynamicRotation 专用参数
    parser.add_argument("--min_hold_days", type=int, help="最小持仓天数")
    parser.add_argument("--max_hold_days", type=int, help="最大持仓天数")
    parser.add_argument("--trailing_stop_pct", type=float, help="移动止损比例")
    parser.add_argument("--enable_discovery", action="store_true", default=None,
                        help="启用主动选股")
    parser.add_argument("--no_discovery", action="store_true", help="禁用主动选股")

    parser.add_argument("--output_file", type=str, default="", help="输出文件名")
    args = parser.parse_args()

    # 读取配置文件
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "breadfree", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # 参数优先级: CLI > config.yaml > 默认值
    strategy_name = args.strategy or config.get("strategy", "RotationStrategy")
    start_date = args.start_date or config.get("start_date", datetime.now().strftime("%Y%m%d"))
    end_date = args.end_date or config.get("end_date",
                                           (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"))
    initial_cash = args.initial_cash or config.get("initial_cash", 100000.0)
    asset_type = config.get("asset_type", "stock")
    lot_size = config.get("lot_size", 100)
    data_source = config.get("data_source", "akshare")
    tushare_token = config.get("tushare_token", None)
    symbols = list(config.get("etf_pool", {"510300": "沪深300ETF"}).keys())

    strategy_cls = STRATEGY_MAP.get(strategy_name, RotationStrategy)

    print(f"Strategy: {strategy_name}")
    print(f"Start Date: {start_date}, End Date: {end_date}, Initial Cash: {initial_cash}")
    print(f"Data Source: {data_source}")
    print(f"Symbols: {symbols}")

    # 构建策略超参数 (BenchmarkStrategy 不需要)
    strategy_params = {}
    if strategy_name != "BenchmarkStrategy":
        param_keys = [
            "lookback_period", "hold_period", "top_n",
            "bias_n", "momentum_day", "slope_n", "rebalance_threshold",
        ]
        for key in param_keys:
            val = getattr(args, key)
            if val is not None:
                strategy_params[key] = val
            elif key in config:
                strategy_params[key] = config[key]

        if args.use_efficiency is not None:
            strategy_params["use_efficiency"] = bool(args.use_efficiency)
        elif "use_efficiency" in config:
            strategy_params["use_efficiency"] = config["use_efficiency"]

        # DynamicRotation 专用参数
        if strategy_name == "DynamicRotation":
            dyn_keys = ["min_hold_days", "max_hold_days", "trailing_stop_pct"]
            for key in dyn_keys:
                val = getattr(args, key, None)
                if val is not None:
                    strategy_params[key] = val
            if args.no_discovery:
                strategy_params["enable_discovery"] = False
            elif args.enable_discovery:
                strategy_params["enable_discovery"] = True

    if strategy_params:
        print("Hyperparameters:")
        for k, v in strategy_params.items():
            print(f"  {k}: {v}")

    print(f"Running backtest with {strategy_cls.__name__}...")

    engine = BacktestEngine(
        strategy_cls=strategy_cls,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        asset_type=asset_type,
        lot_size=lot_size,
        data_source=data_source,
        tushare_token=tushare_token,
        **strategy_params,
    )
    engine.run()

    try:
        if args.output_file != "":
            engine.plot_results_html()
    except Exception as e:
        print(f"Could not plot results: {e}")


if __name__ == "__main__":
    main()
