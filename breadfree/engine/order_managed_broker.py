"""
BreadFree OrderManagedBroker - Risk-Checked Broker Proxy

A transparent proxy that wraps an existing BrokerAdapter and routes all
buy/sell calls through the OrderManager + RiskManager pipeline.

This solves the key integration gap: existing strategies call broker.buy()/sell()
directly, bypassing risk checks. By replacing the broker reference with this proxy,
strategies get automatic pre-trade risk checks, order lifecycle management, and
event-driven audit logging — without any code changes to the strategy itself.

Usage in LiveEngine:
    real_broker = Broker(initial_cash=100000)
    order_manager = OrderManager(broker=real_broker, risk_manager=risk_mgr, event_bus=bus)
    proxy = OrderManagedBroker(real_broker, order_manager)
    strategy = SomeStrategy(proxy)  # strategy sees a normal broker interface
"""

from typing import Dict
from datetime import datetime

from .broker_adapter import BrokerAdapter
from .order_manager import OrderManager
from .models import Signal, Direction, OrderType, OrderStatus, Account
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OrderManagedBroker(BrokerAdapter):
    """
    Proxy broker that intercepts buy/sell and submits them as Signals
    through the OrderManager (which applies RiskManager checks).

    All property accesses (cash, positions, etc.) are delegated to the
    underlying broker so strategies see accurate state.
    """

    def __init__(self, underlying: BrokerAdapter, order_manager: OrderManager):
        """
        :param underlying: The real broker (simulated or live)
        :param order_manager: OMS with risk manager attached
        """
        self._underlying = underlying
        self._order_manager = order_manager

    # ──────────────────────────────────────────
    # Delegated properties
    # ──────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._underlying.cash

    @cash.setter
    def cash(self, value: float):
        if hasattr(self._underlying, 'cash') and isinstance(
                type(self._underlying).cash, property):
            # If underlying has a setter
            self._underlying.cash = value
        else:
            self._underlying._cash = value

    @property
    def positions(self) -> dict:
        return self._underlying.positions

    @positions.setter
    def positions(self, value: dict):
        if hasattr(self._underlying, 'positions') and isinstance(
                type(self._underlying).positions, property):
            self._underlying.positions = value
        else:
            self._underlying._positions = value

    @property
    def commission_rate(self) -> float:
        return self._underlying.commission_rate

    @commission_rate.setter
    def commission_rate(self, value: float):
        if hasattr(self._underlying, 'commission_rate') and isinstance(
                type(self._underlying).commission_rate, property):
            self._underlying.commission_rate = value
        else:
            self._underlying._commission_rate = value

    @property
    def initial_cash(self) -> float:
        return self._underlying.initial_cash

    @initial_cash.setter
    def initial_cash(self, value: float):
        if hasattr(self._underlying, 'initial_cash') and isinstance(
                type(self._underlying).initial_cash, property):
            self._underlying.initial_cash = value
        else:
            self._underlying._initial_cash = value

    # Delegate commonly accessed attributes
    @property
    def equity_curve(self):
        return getattr(self._underlying, 'equity_curve', [])

    @property
    def closed_trades(self):
        return getattr(self._underlying, 'closed_trades', [])

    @property
    def transaction_history(self):
        return getattr(self._underlying, 'transaction_history', [])

    @property
    def current_equity(self):
        return getattr(self._underlying, 'current_equity', 0.0)

    # ──────────────────────────────────────────
    # Core: buy/sell routed through OMS
    # ──────────────────────────────────────────

    def buy(self, date, symbol: str, price: float, quantity: int) -> bool:
        """
        Create a BUY Signal and submit through OrderManager.

        The OrderManager will:
        1. Run pre-trade risk checks (position limits, cash, etc.)
        2. Create an Order and execute via the underlying broker
        3. Publish events (ORDER, TRADE) for audit/persistence
        """
        signal = Signal(
            symbol=symbol,
            direction=Direction.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            reason="strategy_buy",
            timestamp=datetime.now() if not isinstance(date, datetime) else date,
        )

        order = self._order_manager.submit_signal(signal, date=date)

        if order is None:
            logger.warning(f"[OrderManagedBroker] BUY {symbol} x{quantity} - no order created")
            return False

        if order.status == OrderStatus.FILLED:
            return True
        else:
            logger.info(f"[OrderManagedBroker] BUY {symbol} x{quantity} -> {order.status.value}: "
                       f"{order.reject_reason}")
            return False

    def sell(self, date, symbol: str, price: float, quantity: int) -> bool:
        """
        Create a SELL Signal and submit through OrderManager.
        """
        signal = Signal(
            symbol=symbol,
            direction=Direction.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            reason="strategy_sell",
            timestamp=datetime.now() if not isinstance(date, datetime) else date,
        )

        order = self._order_manager.submit_signal(signal, date=date)

        if order is None:
            logger.warning(f"[OrderManagedBroker] SELL {symbol} x{quantity} - no order created")
            return False

        if order.status == OrderStatus.FILLED:
            return True
        else:
            logger.info(f"[OrderManagedBroker] SELL {symbol} x{quantity} -> {order.status.value}: "
                       f"{order.reject_reason}")
            return False

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Delegate to underlying broker."""
        return self._underlying.get_total_equity(current_prices)

    def get_account(self) -> Account:
        """Delegate to underlying broker."""
        return self._underlying.get_account()

    # ──────────────────────────────────────────
    # Direct access (for advanced use cases)
    # ──────────────────────────────────────────

    @property
    def underlying(self) -> BrokerAdapter:
        """Access the underlying broker directly (e.g., for recording equity)."""
        return self._underlying

    @property
    def order_manager(self) -> OrderManager:
        """Access the order manager."""
        return self._order_manager
