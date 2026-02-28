"""
BreadFree 定时邮件报告调度器

每天东八区 08:30 自动生成并发送:
    1. Top-N 标的的最新因子排名 & 决策信号
    2. RotationStrategy 投资组合收益曲线

用法:
    # 后台常驻 (推荐配合 systemd / supervisor / nohup)
    uv run python daily_report_scheduler.py

    # 立即发送一次 (测试)
    uv run python daily_report_scheduler.py --now

环境变量 (.env):
    SMTP_USER=xxx@qq.com
    SMTP_PASSWORD=xxx           # QQ 邮箱授权码
    REPORT_RECIPIENTS=a@qq.com,b@qq.com
"""

import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import schedule
from dotenv import load_dotenv

load_dotenv()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breadfree.utils.config import get_config
from breadfree.utils.logger import get_logger
from breadfree.report.daily_report import generate_and_send_report

logger = get_logger("scheduler")
TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _job():
    now = datetime.now(TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[Scheduler] 触发定时任务 @ {now}")
    try:
        generate_and_send_report()
    except Exception as e:
        logger.error(f"[Scheduler] 报告生成异常: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="BreadFree 定时报告调度器")
    parser.add_argument("--now", action="store_true", help="立即执行一次 (不进入调度循环)")
    parser.add_argument("--time", type=str, default=None,
                        help="覆盖发送时间 (HH:MM), 默认读 config")
    args = parser.parse_args()

    if args.now:
        logger.info("[Scheduler] --now 模式, 立即执行一次")
        generate_and_send_report()
        return

    cfg = get_config().get("daily_report", {})
    send_time = args.time or cfg.get("schedule_time", "08:30")
    tz_name = cfg.get("timezone", "Asia/Shanghai")

    schedule.every().day.at(send_time, tz_name).do(_job)
    logger.info(f"[Scheduler] 已注册定时任务: 每日 {send_time} ({tz_name})")
    tz = ZoneInfo(tz_name)
    logger.info(f"[Scheduler] 当前时间: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
