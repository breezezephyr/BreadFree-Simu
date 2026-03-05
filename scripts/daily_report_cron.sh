#!/usr/bin/env bash
# BreadFree 每日报告触发脚本
# 由 pm2 cron 在每天 16:00 (东八区) 执行

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/daily_report_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting daily report..." | tee -a "$LOG_FILE"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

cd "$PROJECT_DIR"

/usr/local/bin/uv run python -m breadfree.report.daily_report 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daily report finished." | tee -a "$LOG_FILE"
