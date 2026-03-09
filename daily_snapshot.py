"""
daily_snapshot.py — 选股择时数据独立采集脚本

职责:
    每日（或按需）计算固定池 25 标的的因子得分 + 全市场发现扫描，
    将结果写入 breadfree.db 和 JSON 文件，供下游脚本/仪表盘消费。

    与 daily_report_scheduler.py 的区别:
        - 本脚本只做数据计算与持久化，不发邮件，速度更快
        - 可独立运行，也可与报告调度器并行运行
        - 适合作为纯数据采集的 cron 任务

用法:
    # 计算今日快照并写库
    uv run python daily_snapshot.py

    # 指定日期
    uv run python daily_snapshot.py --date 20260306

    # 查询最新选股结果（不计算，只读库）
    uv run python daily_snapshot.py --query

    # 查询历史调仓记录（最近 30 天）
    uv run python daily_snapshot.py --history 30

    # 作为守护进程在每日 08:25（早于报告发送）自动运行
    uv run python daily_snapshot.py --daemon
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _run(date_str: str = None):
    from breadfree.data.snapshot import run_snapshot
    result = run_snapshot(trade_date=date_str)
    if "error" in result:
        print(f"[ERROR] 快照计算失败: {result['error']}")
        return False

    trade_date = result.get("trade_date", "")
    print(f"\n✅ 快照写入完成 [{trade_date}]")
    print(f"   因子记录: {result.get('factors_saved', 0)} 条")
    print(f"   发现记录: {result.get('discovery_saved', 0)} 条")
    print(f"   调仓记录: {'✓' if result.get('rebalance_saved') else '✗'}")

    # 展示 latest.json 路径
    import os
    latest = os.path.join("breadfree", "data", "cache", "snapshot", "latest.json")
    if os.path.exists(latest):
        print(f"\n📄 JSON 快照: {latest}")
    return True


def _query(top_n: int = None):
    from breadfree.data.snapshot import DailySnapshotWriter
    writer = DailySnapshotWriter()

    print("\n===== 最新选股因子快照 =====")
    factors = writer.db.get_latest_factors()
    if not factors:
        print("（暂无数据，请先运行: uv run python daily_snapshot.py）")
        return

    import os
    latest = os.path.join("breadfree", "data", "cache", "snapshot", "latest.json")
    if os.path.exists(latest):
        with open(latest, encoding="utf-8") as f:
            data = json.load(f)
        print(f"数据截止日: {data.get('trade_date')}")
        print(f"Top-{data.get('top_n')} 选股结果:")
        for s in data.get("selection", []):
            mark = "★" if s.get("is_selected") else " "
            print(f"  [{s.get('pool_rank'):>2}] {s['symbol']} {s.get('name', ''):8s} "
                  f"效率={s.get('efficiency', 0):.3f}  动量={s.get('momentum', 0)*100:+.1f}%  "
                  f"R²={s.get('r2', 0):.2f} {mark}")

        print(f"\n全池排名 (共 {len(data.get('all_factors', []))} 条):")
        print(f"{'排名':>4}  {'代码':>8}  {'名称':12}  {'效率分':>8}  {'动量':>8}  {'R²':>6}  {'入选'}")
        print("-" * 70)
        for s in data.get("all_factors", []):
            flag = "✓" if s.get("is_selected") else ""
            print(f"{s.get('pool_rank'):>4}  {s['symbol']:>8}  {s.get('name', ''):12s}  "
                  f"{s.get('efficiency', 0):>8.3f}  {s.get('momentum', 0)*100:>+7.1f}%  "
                  f"{s.get('r2', 0):>6.2f}  {flag}")

        disc = data.get("discovery", {})
        if disc.get("total_discovered", 0) > 0:
            stale_note = " [过期缓存]" if disc.get("is_stale_cache") else ""
            print(f"\n🔍 全市场发现 {disc['total_discovered']} 个标的{stale_note}:")
            for d in disc.get("top_discoveries", [])[:5]:
                print(f"  {d.get('symbol')} {d.get('name', ''):8s} "
                      f"效率={d.get('efficiency', 0):.3f}")
    else:
        # 直接查库
        print(f"{'排名':>4}  {'代码':>8}  {'名称':12}  {'效率分':>8}  {'入选'}")
        for f in factors:
            flag = "✓" if f.get("is_selected") else ""
            print(f"{f.get('pool_rank', '?'):>4}  {f.get('symbol', ''):>8}  "
                  f"{f.get('name', ''):12s}  {f.get('efficiency', 0):>8.3f}  {flag}")


def _history(days: int = 30):
    from breadfree.data.snapshot import DailySnapshotWriter
    writer = DailySnapshotWriter()
    logs = writer.get_rebalance_history(days=days)
    if not logs:
        print(f"（最近 {days} 天无调仓记录）")
        return

    print(f"\n===== 最近 {days} 天调仓记录 =====")
    for log in logs:
        selected = log.get("selected", [])
        symbols_str = ", ".join(
            f"{s['symbol']}({s.get('efficiency', 0):.2f})" for s in selected
        )
        print(f"[{log.get('trade_date')}] Top-{log.get('top_n')} | {symbols_str}")


def _daemon(schedule_time: str = "08:25"):
    try:
        import schedule
    except ImportError:
        print("需要安装 schedule: uv add schedule")
        sys.exit(1)

    def _job():
        print(f"\n[{datetime.now(TZ_SHANGHAI).strftime('%H:%M:%S')}] 执行每日快照...")
        _run()

    schedule.every().day.at(schedule_time, "Asia/Shanghai").do(_job)
    print(f"✅ 快照守护进程已启动，每日 {schedule_time} (Asia/Shanghai) 自动运行")
    print("   按 Ctrl+C 退出")
    while True:
        schedule.run_pending()
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(
        description="BreadFree 选股择时数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, default=None,
                        help="指定截止日期 YYYYMMDD（默认: 最近交易日）")
    parser.add_argument("--query", action="store_true",
                        help="查询最新选股结果（不重新计算）")
    parser.add_argument("--history", type=int, metavar="DAYS",
                        help="查询最近 N 天调仓历史")
    parser.add_argument("--daemon", action="store_true",
                        help="作为守护进程每日自动运行")
    parser.add_argument("--time", type=str, default="08:25",
                        help="守护进程运行时间，格式 HH:MM（默认 08:25）")
    args = parser.parse_args()

    if args.daemon:
        _daemon(args.time)
    elif args.query:
        _query()
    elif args.history is not None:
        _history(args.history)
    else:
        ok = _run(args.date)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
