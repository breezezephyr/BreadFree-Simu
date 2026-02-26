"""
BreadFree AuditLogger - Trade & Decision Audit Trail

Records every significant event with full context for:
- Regulatory compliance (trade audit trail)
- LLM decision review (reasoning + signals)
- Post-mortem analysis (what happened and why)

All audit entries are persisted to a dedicated SQLite table and
optionally streamed to the EventBus for real-time monitoring.

Usage:
    audit = AuditLogger("live_trading.db")
    audit.log_order(order, context="Strategy rebalance")
    audit.log_risk_event("Max drawdown hit", details={...})
    audit.log_strategy_decision(strategy_name, signals, reasoning)
"""

import json
import sqlite3
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AuditCategory:
    """Audit log categories."""
    ORDER = "ORDER"
    TRADE = "TRADE"
    RISK = "RISK"
    STRATEGY = "STRATEGY"
    SYSTEM = "SYSTEM"
    LLM = "LLM"
    GATEWAY = "GATEWAY"


class AuditLogger:
    """
    Structured audit logger for live trading.

    Stores audit entries in an SQLite table alongside the live trading
    database, providing a complete and queryable audit trail.
    """

    def __init__(self, db_path: str = "live_trading.db"):
        self.db_path = db_path
        self._init_table()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_table(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details TEXT DEFAULT '',
                    level TEXT DEFAULT 'INFO'
                );
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_category
                    ON audit_log(category);
            """)

    # ──────────────────────────────────────────
    # Core write
    # ──────────────────────────────────────────

    def log(self, category: str, action: str, summary: str,
            details: Any = None, level: str = "INFO"):
        """
        Write an audit entry.

        :param category: AuditCategory value (ORDER, TRADE, RISK, etc.)
        :param action: Verb describing what happened (CREATED, REJECTED, etc.)
        :param summary: One-line human-readable description
        :param details: Optional dict/object serialized to JSON
        :param level: INFO / WARNING / ERROR
        """
        details_str = ""
        if details is not None:
            try:
                details_str = json.dumps(details, ensure_ascii=False, default=str)
            except Exception:
                details_str = str(details)

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO audit_log (timestamp, category, action, summary, details, level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                category,
                action,
                summary,
                details_str,
                level,
            ))

        # Also log to file/console for real-time visibility
        logger.info(f"[Audit:{category}:{action}] {summary}")

    # ──────────────────────────────────────────
    # Convenience methods
    # ──────────────────────────────────────────

    def log_order(self, order, context: str = ""):
        """Log an order lifecycle event."""
        details = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "direction": order.direction.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "reject_reason": order.reject_reason,
            "context": context,
        }
        level = "WARNING" if order.status.value == "REJECTED" else "INFO"
        summary = (f"{order.direction.value} {order.symbol} x{order.quantity} "
                  f"@ {order.price or 'MKT'} -> {order.status.value}")
        if order.reject_reason:
            summary += f" ({order.reject_reason})"
        self.log(AuditCategory.ORDER, order.status.value, summary, details, level)

    def log_trade(self, trade):
        """Log a trade execution."""
        details = {
            "trade_id": trade.trade_id,
            "order_id": trade.order_id,
            "symbol": trade.symbol,
            "direction": trade.direction.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "commission": trade.commission,
        }
        summary = (f"FILL {trade.direction.value} {trade.symbol} "
                  f"x{trade.quantity} @ {trade.price:.4f} "
                  f"(commission={trade.commission:.2f})")
        self.log(AuditCategory.TRADE, "FILLED", summary, details)

    def log_risk_event(self, event_description: str, details: dict = None):
        """Log a risk management event."""
        self.log(AuditCategory.RISK, "CHECK", event_description, details, "WARNING")

    def log_strategy_decision(self, strategy_name: str, signals: list = None,
                             reasoning: str = ""):
        """Log a strategy decision (especially useful for LLM agents)."""
        details = {
            "strategy": strategy_name,
            "signals_count": len(signals) if signals else 0,
            "signals": signals[:20] if signals else [],  # Limit to 20
            "reasoning": reasoning[:2000],  # Limit reasoning length
        }
        summary = f"{strategy_name}: {len(signals or [])} signals"
        self.log(AuditCategory.STRATEGY, "DECISION", summary, details)

    def log_llm_call(self, model: str, prompt_summary: str, response_summary: str,
                    tokens_used: int = 0, latency_ms: int = 0):
        """Log an LLM API call for cost tracking and debugging."""
        details = {
            "model": model,
            "prompt": prompt_summary[:500],
            "response": response_summary[:500],
            "tokens": tokens_used,
            "latency_ms": latency_ms,
        }
        summary = f"LLM:{model} tokens={tokens_used} latency={latency_ms}ms"
        self.log(AuditCategory.LLM, "CALL", summary, details)

    def log_system_event(self, action: str, description: str, level: str = "INFO"):
        """Log a system-level event (startup, shutdown, reconnect)."""
        self.log(AuditCategory.SYSTEM, action, description, level=level)

    def log_gateway_event(self, gateway: str, action: str, description: str,
                         level: str = "INFO"):
        """Log a gateway event (connect, disconnect, error)."""
        details = {"gateway": gateway}
        self.log(AuditCategory.GATEWAY, action, f"[{gateway}] {description}",
                details, level)

    # ──────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────

    def get_recent(self, limit: int = 100, category: str = None) -> List[dict]:
        """Get recent audit log entries."""
        with self._get_conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE category = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (category, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    def get_by_date(self, target_date: date = None) -> List[dict]:
        """Get all audit entries for a given date."""
        d = (target_date or date.today()).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE timestamp >= ? "
                "AND timestamp < date(?, '+1 day') ORDER BY id",
                (d, d)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_warnings_and_errors(self, limit: int = 50) -> List[dict]:
        """Get recent warnings and errors."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE level IN ('WARNING', 'ERROR') "
                "ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def count_by_category(self, target_date: date = None) -> Dict[str, int]:
        """Count audit entries by category for a given date."""
        d = (target_date or date.today()).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM audit_log "
                "WHERE timestamp >= ? AND timestamp < date(?, '+1 day') "
                "GROUP BY category",
                (d, d)
            ).fetchall()
            return {row["category"]: row["cnt"] for row in rows}
