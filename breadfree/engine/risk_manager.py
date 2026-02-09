"""
BreadFree Risk Manager

Pre-trade and post-trade risk checks to protect capital.

Rules are configurable via config dict (loaded from YAML).

Design reference:
- vnpy: RiskManagerEngine with configurable rules
- nautilus_trader: RiskEngine with position limits
"""

from typing import Tuple, Set, Optional
from datetime import datetime, date as date_type

from .models import Signal, Order, Direction
from .broker_adapter import BrokerAdapter
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Risk management system with configurable rules.

    Pre-trade checks are performed before every order submission.
    Post-trade updates track daily trading activity.

    Default rules (all configurable):
    - max_position_pct: Maximum single position as % of equity (default: 0.40)
    - max_order_value: Maximum single order value (default: 500,000)
    - max_daily_trades: Maximum trades per day (default: 200)
    - blacklist: Set of symbols that cannot be traded
    - check_t1: Enable A-share T+1 sell restriction check (default: True)
    - max_drawdown_pct: Maximum drawdown before locking (default: 0.20)
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        # Configurable limits
        self.max_position_pct = config.get("max_position_pct", 0.40)
        self.max_order_value = config.get("max_order_value", 500_000)
        self.max_daily_trades = config.get("max_daily_trades", 200)
        self.blacklist: Set[str] = set(config.get("blacklist", []))
        self.whitelist: Set[str] = set(config.get("whitelist", []))  # Empty = all allowed
        self.check_t1 = config.get("check_t1", True)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.20)

        # State tracking
        self._daily_trade_count = 0
        self._current_date: Optional[date_type] = None
        self._buy_dates: dict = {}  # symbol -> last buy date (for T+1 check)
        self._is_locked = False     # Lock trading when max drawdown is hit
        self._peak_equity = 0.0

        # Statistics
        self.rejected_count = 0
        self.approved_count = 0

    # ──────────────────────────────────────────
    # Pre-trade check
    # ──────────────────────────────────────────

    def pre_trade_check(self, signal: Signal, broker: BrokerAdapter) -> Tuple[bool, str]:
        """
        Run all pre-trade risk checks on a signal.

        :param signal: The trading signal to check
        :param broker: Current broker state
        :return: (passed: bool, reason: str)
        """
        # 0. Check if trading is locked (max drawdown protection)
        if self._is_locked:
            self.rejected_count += 1
            return False, "Trading locked: max drawdown protection active"

        # 1. Blacklist check
        if signal.symbol in self.blacklist:
            self.rejected_count += 1
            return False, f"{signal.symbol} is in blacklist"

        # 2. Whitelist check (if configured)
        if self.whitelist and signal.symbol not in self.whitelist:
            self.rejected_count += 1
            return False, f"{signal.symbol} is not in whitelist"

        # 3. Daily trade count limit
        self._update_daily_counter(signal.timestamp)
        if self._daily_trade_count >= self.max_daily_trades:
            self.rejected_count += 1
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # 4. Single order value limit
        price = signal.price or 0.0
        order_value = price * signal.quantity
        if order_value > self.max_order_value:
            self.rejected_count += 1
            return False, (f"Order value {order_value:.2f} exceeds limit "
                         f"{self.max_order_value:.2f}")

        # 5. Direction-specific checks
        if signal.direction == Direction.BUY:
            return self._check_buy(signal, broker)
        elif signal.direction == Direction.SELL:
            return self._check_sell(signal, broker)

        self.rejected_count += 1
        return False, f"Unknown direction: {signal.direction}"

    def _check_buy(self, signal: Signal, broker: BrokerAdapter) -> Tuple[bool, str]:
        """Buy-specific risk checks."""
        price = signal.price or 0.0

        # Cash sufficiency
        needed_cash = price * signal.quantity * (1 + broker.commission_rate)
        if broker.cash < needed_cash:
            self.rejected_count += 1
            return False, (f"Insufficient cash: need {needed_cash:.2f}, "
                         f"available {broker.cash:.2f}")

        # Single position concentration limit
        total_equity = broker.current_equity if hasattr(broker, 'current_equity') else broker.cash
        if total_equity > 0:
            buy_value = price * signal.quantity
            # Include existing position value
            existing_value = 0
            if signal.symbol in broker.positions:
                pos = broker.positions[signal.symbol]
                existing_value = pos.quantity * price
            new_position_pct = (existing_value + buy_value) / total_equity
            if new_position_pct > self.max_position_pct:
                self.rejected_count += 1
                return False, (f"Position concentration {new_position_pct:.2%} "
                             f"exceeds limit {self.max_position_pct:.2%}")

        self.approved_count += 1
        return True, "OK"

    def _check_sell(self, signal: Signal, broker: BrokerAdapter) -> Tuple[bool, str]:
        """Sell-specific risk checks."""
        # Position existence check
        if signal.symbol not in broker.positions:
            self.rejected_count += 1
            return False, f"No position in {signal.symbol}"

        pos = broker.positions[signal.symbol]
        if pos.quantity < signal.quantity:
            self.rejected_count += 1
            return False, (f"Insufficient position: want to sell {signal.quantity}, "
                         f"holding {pos.quantity}")

        # T+1 check (A-share: cannot sell shares bought today)
        if self.check_t1 and signal.symbol in self._buy_dates:
            last_buy = self._buy_dates[signal.symbol]
            signal_date = signal.timestamp.date() if signal.timestamp else None
            if signal_date and last_buy == signal_date:
                self.rejected_count += 1
                return False, f"T+1 restriction: {signal.symbol} was bought today"

        self.approved_count += 1
        return True, "OK"

    # ──────────────────────────────────────────
    # Post-trade update
    # ──────────────────────────────────────────

    def post_trade_update(self, order: Order):
        """
        Update risk state after a trade is executed.

        :param order: The executed order
        """
        self._daily_trade_count += 1

        # Track buy dates for T+1
        if order.direction == Direction.BUY:
            if order.create_time:
                self._buy_dates[order.symbol] = order.create_time.date()

    def update_equity(self, current_equity: float):
        """
        Update equity tracking for drawdown protection.
        Should be called at the end of each trading day.

        :param current_equity: Current total equity
        """
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                logger.warning(
                    f"Max drawdown protection triggered! "
                    f"Drawdown: {drawdown:.2%} >= {self.max_drawdown_pct:.2%}. "
                    f"Trading locked."
                )
                self._is_locked = True

    def unlock(self):
        """Manually unlock trading (after drawdown recovery or manual override)."""
        self._is_locked = False
        logger.info("Trading unlocked by manual override.")

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _update_daily_counter(self, timestamp):
        """Reset daily trade counter if a new day has started."""
        if timestamp is None:
            return

        current = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if isinstance(current, datetime):
            current = current.date()

        if self._current_date != current:
            self._current_date = current
            self._daily_trade_count = 0

    def reset_daily(self):
        """Reset daily state (called at start of each trading day)."""
        self._daily_trade_count = 0

    # ──────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get risk manager statistics."""
        return {
            "approved": self.approved_count,
            "rejected": self.rejected_count,
            "daily_trades_today": self._daily_trade_count,
            "is_locked": self._is_locked,
            "peak_equity": self._peak_equity,
        }
