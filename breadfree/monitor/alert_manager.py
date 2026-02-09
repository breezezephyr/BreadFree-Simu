"""
BreadFree AlertManager - Multi-channel Alert & Notification

Sends real-time alerts via:
- WeChat Work (企业微信) Robot Webhook
- DingTalk (钉钉) Robot Webhook
- Console / Log fallback

Alert levels:
- CRITICAL: System crash, connection lost, capital anomaly -> all channels
- WARNING:  Order rejected, risk triggered, LLM failure -> instant message
- INFO:     Daily summary, strategy signals -> scheduled push

Usage:
    alert = AlertManager(config)
    alert.send("Order rejected: insufficient cash", level=AlertLevel.WARNING)
    alert.send_daily_summary(account, trades)
"""

import json
import time
import urllib.request
import urllib.error
from enum import Enum
from datetime import datetime, date
from typing import Optional, List, Dict
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    level: AlertLevel
    message: str
    channel: str
    success: bool
    timestamp: datetime
    error: str = ""


class AlertManager:
    """
    Multi-channel alert manager for live trading monitoring.

    Supports WeChat Work and DingTalk robot webhooks.
    Falls back to console logging if no webhook is configured.

    Rate limiting: max 20 messages per minute per channel to avoid spam.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        :param config: Alert configuration dict

        Expected config:
        {
            "wechat_webhook": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
            "dingtalk_webhook": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
            "enabled": true,
            "min_level": "WARNING",   # Only send alerts >= this level
            "rate_limit_per_min": 20,
        }
        """
        config = config or {}
        self._wechat_webhook = config.get("wechat_webhook", "")
        self._dingtalk_webhook = config.get("dingtalk_webhook", "")
        self._enabled = config.get("enabled", True)
        self._min_level_str = config.get("min_level", "INFO")
        self._rate_limit = config.get("rate_limit_per_min", 20)

        # Level priority for filtering
        self._level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.CRITICAL: 2,
        }
        self._min_level = AlertLevel[self._min_level_str] if self._min_level_str in AlertLevel.__members__ else AlertLevel.INFO

        # Rate limiting state
        self._send_timestamps: List[float] = []

        # History
        self._alert_history: List[AlertRecord] = []

        # Log configuration
        channels = []
        if self._wechat_webhook:
            channels.append("WeChat")
        if self._dingtalk_webhook:
            channels.append("DingTalk")
        if not channels:
            channels.append("Console-only")
        logger.info(f"[AlertManager] Initialized: channels={channels}, "
                    f"min_level={self._min_level.value}, enabled={self._enabled}")

    # ──────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO,
             title: str = "BreadFree Alert") -> bool:
        """
        Send an alert message to all configured channels.

        :param message: Alert message body
        :param level: Alert severity level
        :param title: Alert title (used in formatted messages)
        :return: True if at least one channel succeeded
        """
        if not self._enabled:
            return False

        # Level filtering
        if self._level_priority[level] < self._level_priority[self._min_level]:
            return False

        # Rate limiting
        if not self._check_rate_limit():
            logger.warning("[AlertManager] Rate limit exceeded, alert dropped")
            return False

        # Format message
        formatted = self._format_message(title, message, level)

        # Always log to console
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
        }.get(level, logger.info)
        log_func(f"[Alert:{level.value}] {message}")

        # Send to channels
        any_success = False

        if self._wechat_webhook:
            success = self._send_wechat(formatted)
            self._record(level, message, "WeChat", success)
            any_success = any_success or success

        if self._dingtalk_webhook:
            success = self._send_dingtalk(title, formatted, level)
            self._record(level, message, "DingTalk", success)
            any_success = any_success or success

        # If no webhook configured, console log counts as success
        if not self._wechat_webhook and not self._dingtalk_webhook:
            self._record(level, message, "Console", True)
            any_success = True

        return any_success

    def send_critical(self, message: str, title: str = "CRITICAL ALERT"):
        """Convenience: send a CRITICAL alert."""
        return self.send(message, level=AlertLevel.CRITICAL, title=title)

    def send_warning(self, message: str, title: str = "Warning"):
        """Convenience: send a WARNING alert."""
        return self.send(message, level=AlertLevel.WARNING, title=title)

    def send_info(self, message: str, title: str = "Info"):
        """Convenience: send an INFO alert."""
        return self.send(message, level=AlertLevel.INFO, title=title)

    # ──────────────────────────────────────────
    # Structured alerts
    # ──────────────────────────────────────────

    def send_daily_summary(self, account_info: dict, trades: List[dict],
                          risk_stats: dict = None):
        """
        Send end-of-day trading summary.

        :param account_info: {"total_equity": ..., "available_cash": ..., "total_pnl": ...}
        :param trades: List of trade records from today
        :param risk_stats: Risk manager statistics
        """
        today_str = date.today().strftime("%Y-%m-%d")

        lines = [
            f"--- {today_str} Daily Summary ---",
            "",
            f"Total Equity: {account_info.get('total_equity', 0):.2f}",
            f"Available Cash: {account_info.get('available_cash', 0):.2f}",
            f"P&L Today: {account_info.get('total_pnl', 0):.2f}",
            f"Trades Today: {len(trades)}",
        ]

        if trades:
            lines.append("")
            lines.append("Recent Trades:")
            for t in trades[:10]:  # Max 10 trades
                direction = t.get('direction', '?')
                symbol = t.get('symbol', '?')
                qty = t.get('quantity', 0)
                price = t.get('price', 0)
                lines.append(f"  {direction} {symbol} x{qty} @ {price:.3f}")

        if risk_stats:
            lines.append("")
            lines.append(f"Risk: approved={risk_stats.get('approved', 0)}, "
                        f"rejected={risk_stats.get('rejected', 0)}, "
                        f"locked={risk_stats.get('is_locked', False)}")

        message = "\n".join(lines)
        return self.send(message, level=AlertLevel.INFO, title=f"Daily Summary {today_str}")

    def send_order_alert(self, order_info: dict):
        """Send alert for important order events (rejection, large orders)."""
        status = order_info.get("status", "UNKNOWN")
        symbol = order_info.get("symbol", "?")
        direction = order_info.get("direction", "?")
        quantity = order_info.get("quantity", 0)

        if status == "REJECTED":
            reason = order_info.get("reject_reason", "unknown")
            msg = f"Order REJECTED: {direction} {symbol} x{quantity} - {reason}"
            return self.send(msg, level=AlertLevel.WARNING, title="Order Rejected")

        return False

    def send_risk_alert(self, risk_event: str):
        """Send alert for risk management events."""
        return self.send(risk_event, level=AlertLevel.WARNING, title="Risk Alert")

    def send_system_alert(self, error: str):
        """Send alert for system-level errors (crashes, disconnects)."""
        return self.send(error, level=AlertLevel.CRITICAL, title="System Error")

    # ──────────────────────────────────────────
    # Channel implementations
    # ──────────────────────────────────────────

    def _send_wechat(self, message: str) -> bool:
        """Send message via WeChat Work robot webhook."""
        payload = {
            "msgtype": "text",
            "text": {"content": message}
        }
        return self._post_json(self._wechat_webhook, payload, "WeChat")

    def _send_dingtalk(self, title: str, message: str, level: AlertLevel) -> bool:
        """Send message via DingTalk robot webhook."""
        # DingTalk supports markdown for richer formatting
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": f"### {title}\n\n{message}",
            }
        }
        return self._post_json(self._dingtalk_webhook, payload, "DingTalk")

    def _post_json(self, url: str, payload: dict, channel: str) -> bool:
        """Post JSON to a webhook URL."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                # WeChat: errcode == 0; DingTalk: errcode == 0
                errcode = result.get("errcode", -1)
                if errcode == 0:
                    logger.debug(f"[AlertManager] {channel} sent OK")
                    return True
                else:
                    logger.warning(f"[AlertManager] {channel} API error: {result}")
                    return False
        except urllib.error.URLError as e:
            logger.error(f"[AlertManager] {channel} network error: {e}")
            return False
        except Exception as e:
            logger.error(f"[AlertManager] {channel} error: {e}")
            return False

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _format_message(self, title: str, message: str, level: AlertLevel) -> str:
        """Format alert message with metadata."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level.value, "")
        return f"{icon} [{level.value}] {title}\n{now}\n\n{message}"

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._send_timestamps = [t for t in self._send_timestamps if now - t < 60]
        if len(self._send_timestamps) >= self._rate_limit:
            return False
        self._send_timestamps.append(now)
        return True

    def _record(self, level: AlertLevel, message: str, channel: str, success: bool,
                error: str = ""):
        """Record alert in history."""
        self._alert_history.append(AlertRecord(
            level=level,
            message=message[:200],  # Truncate for storage
            channel=channel,
            success=success,
            timestamp=datetime.now(),
            error=error,
        ))
        # Keep last 1000 records
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

    def get_history(self, limit: int = 50) -> List[dict]:
        """Get recent alert history."""
        records = self._alert_history[-limit:]
        return [
            {
                "level": r.level.value,
                "message": r.message,
                "channel": r.channel,
                "success": r.success,
                "timestamp": r.timestamp.isoformat(),
                "error": r.error,
            }
            for r in reversed(records)
        ]
