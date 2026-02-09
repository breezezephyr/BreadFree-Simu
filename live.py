#!/usr/bin/env python3
"""
BreadFree Live Trading Entry Point

Usage:
    # Simulated (paper trading):
    uv run python live.py

    # Simulated with custom config:
    uv run python live.py --config config/live.yaml

    # One-shot (run strategy once, don't wait for schedule):
    uv run python live.py --once

    # Check status only:
    uv run python live.py --status
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="BreadFree 实盘交易系统")
    parser.add_argument(
        "--config", "-c",
        default="config/live.yaml",
        help="配置文件路径 (默认: config/live.yaml)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="立即运行一次策略，不等待调度"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查看引擎状态"
    )
    parser.add_argument(
        "--gateway",
        choices=["simulated", "futu", "qmt"],
        help="覆盖配置中的 gateway 类型"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override gateway if specified
    if args.gateway:
        config["gateway"]["type"] = args.gateway

    # Initialize engine
    from breadfree.engine.live_engine import LiveEngine

    engine = LiveEngine(config)

    if args.status:
        status = engine.get_status()
        print("\n=== BreadFree Live Engine Status ===")
        for key, val in status.items():
            print(f"  {key}: {val}")
        return

    if args.once:
        print("\n=== BreadFree One-Shot Mode ===")
        print(f"策略: {config['strategy']['name']}")
        print(f"标的: {config['symbols']}")
        print("正在执行策略...")

        # Connect
        if engine.gateway and not engine.gateway.is_connected:
            engine.gateway.connect()
            if config.get("symbols"):
                engine.gateway.subscribe(config["symbols"])

        # Run once
        engine.trigger_strategy_now()

        # Print result
        status = engine.get_status()
        print(f"\n完成! OMS 统计: {status['oms_stats']}")
        print(f"风控统计: {status['risk_stats']}")

        engine.stop()
        return

    # Normal mode: start scheduler
    print("\n╔═══════════════════════════════════════╗")
    print("║   BreadFree Live Trading Engine       ║")
    print("╠═══════════════════════════════════════╣")
    print(f"║  Gateway:  {config['gateway']['type']:<26}║")
    print(f"║  Market:   {config.get('market', 'A_SHARE'):<26}║")
    print(f"║  Strategy: {config['strategy']['name']:<26}║")
    print(f"║  Symbols:  {len(config.get('symbols', []))} 只                       ║")
    print("╚═══════════════════════════════════════╝")
    print("\nPress Ctrl+C to stop.\n")

    engine.run()


if __name__ == "__main__":
    main()
