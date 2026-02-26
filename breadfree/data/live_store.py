"""
BreadFree Live Trading State Persistence

Stores live trading state to SQLite for crash recovery and audit:
- Orders: full lifecycle history
- Trades: execution records
- Positions: end-of-day snapshots
- Equity Curve: daily equity tracking
- Engine State: scheduler phase, last run time, etc.

Usage:
    store = LiveTradeStore("live_trading.db")
    store.save_order(order)
    store.save_trade(trade)
    store.save_position_snapshot(date, positions)
    store.save_equity_point(date, equity, cash)

    # Recovery
    orders = store.get_today_orders()
    last_state = store.get_engine_state()
"""

import sqlite3
from datetime import datetime, date
from typing import List, Optional, Dict
from contextlib import contextmanager

from ..engine.models import (
    Order, Trade, Account, PositionInfo,
    Direction, OrderType, OrderStatus,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LiveTradeStore:
    """
    SQLite-based persistence for live trading state.

    Tables:
    - live_orders: Order lifecycle records
    - live_trades: Execution / fill records
    - live_position_snapshots: End-of-day position snapshots
    - live_equity_curve: Daily equity tracking
    - live_engine_state: Key-value store for engine recovery
    """

    def __init__(self, db_path: str = "live_trading.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
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

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS live_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL DEFAULT 0.0,
                    commission REAL DEFAULT 0.0,
                    reject_reason TEXT DEFAULT '',
                    source TEXT DEFAULT '',
                    create_time TEXT NOT NULL,
                    update_time TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_trades (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES live_orders(order_id)
                );

                CREATE TABLE IF NOT EXISTS live_position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    market_price REAL DEFAULT 0.0,
                    market_value REAL DEFAULT 0.0,
                    unrealized_pnl REAL DEFAULT 0.0,
                    UNIQUE(snapshot_date, symbol)
                );

                CREATE TABLE IF NOT EXISTS live_equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_date TEXT NOT NULL UNIQUE,
                    total_equity REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    position_value REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS live_engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_orders_symbol ON live_orders(symbol);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON live_orders(status);
                CREATE INDEX IF NOT EXISTS idx_orders_create_time ON live_orders(create_time);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON live_trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON live_trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_positions_date ON live_position_snapshots(snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_equity_date ON live_equity_curve(record_date);
            """)

    # ──────────────────────────────────────────
    # Orders
    # ──────────────────────────────────────────

    def save_order(self, order: Order):
        """Save or update an order record."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_orders
                (order_id, symbol, direction, order_type, quantity, price,
                 status, filled_quantity, avg_fill_price, commission,
                 reject_reason, source, create_time, update_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.symbol,
                order.direction.value,
                order.order_type.value,
                order.quantity,
                order.price or 0.0,
                order.status.value,
                order.filled_quantity,
                order.avg_fill_price,
                order.commission,
                order.reject_reason,
                order.source,
                order.create_time.isoformat() if order.create_time else datetime.now().isoformat(),
                order.update_time.isoformat() if order.update_time else datetime.now().isoformat(),
            ))

    def get_today_orders(self, today: Optional[date] = None) -> List[dict]:
        """Get all orders from today."""
        today = today or date.today()
        today_str = today.isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_orders WHERE create_time >= ? ORDER BY create_time",
                (today_str,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_orders_by_status(self, status: str) -> List[dict]:
        """Get orders by status."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_orders WHERE status = ? ORDER BY create_time DESC",
                (status,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_orders(self, limit: int = 50) -> List[dict]:
        """Get most recent orders."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_orders ORDER BY create_time DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ──────────────────────────────────────────
    # Trades
    # ──────────────────────────────────────────

    def save_trade(self, trade: Trade):
        """Save a trade execution record."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_trades
                (trade_id, order_id, symbol, direction, quantity, price,
                 commission, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id,
                trade.order_id,
                trade.symbol,
                trade.direction.value,
                trade.quantity,
                trade.price,
                trade.commission,
                trade.timestamp.isoformat() if trade.timestamp else datetime.now().isoformat(),
            ))

    def get_today_trades(self, today: Optional[date] = None) -> List[dict]:
        """Get all trades from today."""
        today = today or date.today()
        today_str = today.isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_trades WHERE timestamp >= ? ORDER BY timestamp",
                (today_str,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_trades_by_symbol(self, symbol: str, limit: int = 100) -> List[dict]:
        """Get trades for a specific symbol."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    # ──────────────────────────────────────────
    # Position snapshots
    # ──────────────────────────────────────────

    def save_position_snapshot(self, snapshot_date: date, positions: List[PositionInfo]):
        """Save end-of-day position snapshot."""
        date_str = snapshot_date.isoformat()
        with self._get_conn() as conn:
            for pos in positions:
                conn.execute("""
                    INSERT OR REPLACE INTO live_position_snapshots
                    (snapshot_date, symbol, quantity, avg_price,
                     market_price, market_value, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    pos.symbol,
                    pos.quantity,
                    pos.avg_price,
                    pos.market_price,
                    pos.market_value,
                    pos.unrealized_pnl,
                ))

    def get_position_snapshot(self, snapshot_date: Optional[date] = None) -> List[dict]:
        """Get position snapshot for a date."""
        date_str = (snapshot_date or date.today()).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM live_position_snapshots WHERE snapshot_date = ?",
                (date_str,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_latest_position_snapshot(self) -> List[dict]:
        """Get the most recent position snapshot."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(snapshot_date) as latest FROM live_position_snapshots"
            ).fetchone()
            if row and row["latest"]:
                return self.get_position_snapshot(date.fromisoformat(row["latest"]))
            return []

    # ──────────────────────────────────────────
    # Equity curve
    # ──────────────────────────────────────────

    def save_equity_point(self, record_date: date, total_equity: float,
                         available_cash: float, position_value: float = 0.0,
                         total_pnl: float = 0.0):
        """Save a daily equity curve data point."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_equity_curve
                (record_date, total_equity, available_cash, position_value, total_pnl)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record_date.isoformat(),
                total_equity,
                available_cash,
                position_value,
                total_pnl,
            ))

    def get_equity_curve(self, start_date: Optional[date] = None,
                        end_date: Optional[date] = None) -> List[dict]:
        """Get equity curve data points."""
        with self._get_conn() as conn:
            query = "SELECT * FROM live_equity_curve"
            params = []
            conditions = []

            if start_date:
                conditions.append("record_date >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("record_date <= ?")
                params.append(end_date.isoformat())

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY record_date"
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ──────────────────────────────────────────
    # Engine state (key-value)
    # ──────────────────────────────────────────

    def save_state(self, key: str, value: str):
        """Save engine state key-value pair."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_engine_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))

    def get_state(self, key: str, default: str = "") -> str:
        """Get engine state value."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM live_engine_state WHERE key = ?",
                (key,)
            ).fetchone()
            return row["value"] if row else default

    def get_all_state(self) -> Dict[str, str]:
        """Get all engine state key-value pairs."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM live_engine_state").fetchall()
            return {row["key"]: row["value"] for row in rows}

    # ──────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────

    def get_summary(self) -> dict:
        """Get a summary of stored data."""
        with self._get_conn() as conn:
            orders_count = conn.execute("SELECT COUNT(*) FROM live_orders").fetchone()[0]
            trades_count = conn.execute("SELECT COUNT(*) FROM live_trades").fetchone()[0]
            snapshots = conn.execute(
                "SELECT COUNT(DISTINCT snapshot_date) FROM live_position_snapshots"
            ).fetchone()[0]
            equity_points = conn.execute("SELECT COUNT(*) FROM live_equity_curve").fetchone()[0]

            # Latest equity
            latest = conn.execute(
                "SELECT total_equity, record_date FROM live_equity_curve "
                "ORDER BY record_date DESC LIMIT 1"
            ).fetchone()

            return {
                "total_orders": orders_count,
                "total_trades": trades_count,
                "position_snapshot_days": snapshots,
                "equity_curve_points": equity_points,
                "latest_equity": latest["total_equity"] if latest else None,
                "latest_equity_date": latest["record_date"] if latest else None,
            }
