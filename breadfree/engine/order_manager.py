"""
BreadFree Order Management System (OMS)

Manages the full lifecycle of orders:
  Signal -> Risk Check -> Order Creation -> Submission -> Fill -> Record

In backtest mode, orders are filled instantly via SimulatedBrokerAdapter.
In live mode, orders go through the Gateway and receive async callbacks.

Design reference:
- vnpy: OrderData / TradeData lifecycle
- nautilus_trader: OrderManager with state machine
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import (
    Signal, Order, Trade, OrderStatus, Direction, OrderType,
    Event, EventType,
)
from .broker_adapter import BrokerAdapter
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Central order management system.

    Responsibilities:
    - Create orders from signals
    - Coordinate with RiskManager for pre-trade checks
    - Submit orders to BrokerAdapter
    - Track order status and fill history
    - Publish events via EventBus (optional)
    """

    def __init__(self, broker: BrokerAdapter, risk_manager=None, event_bus=None):
        """
        :param broker: BrokerAdapter instance (simulated or live)
        :param risk_manager: Optional RiskManager for pre-trade checks
        :param event_bus: Optional EventBus for publishing order/trade events
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.event_bus = event_bus

        # Order tracking
        self._active_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._trade_history: List[Trade] = []

        # Counters
        self._order_counter = 0
        self._trade_counter = 0

    # ──────────────────────────────────────────
    # Order creation & submission
    # ──────────────────────────────────────────

    def submit_signal(self, signal: Signal, date=None) -> Optional[Order]:
        """
        Process a trading signal: risk check -> create order -> execute.

        :param signal: Trading signal from strategy
        :param date: Current date (for backtest compatibility)
        :return: Created Order (or None if rejected)
        """
        # 1. Pre-trade risk check
        if self.risk_manager is not None:
            passed, reason = self.risk_manager.pre_trade_check(signal, self.broker)
            if not passed:
                logger.warning(f"Order rejected by RiskManager: {signal.symbol} "
                             f"{signal.direction.value} {signal.quantity} - {reason}")
                rejected_order = self._create_order(signal)
                rejected_order.status = OrderStatus.REJECTED
                rejected_order.reject_reason = reason
                self._order_history.append(rejected_order)
                self._publish_event(EventType.ORDER, rejected_order)
                return rejected_order

        # 2. Create order
        order = self._create_order(signal)
        self._active_orders[order.order_id] = order
        self._publish_event(EventType.ORDER, order)

        # 3. Execute via broker
        success = self._execute_order(order, date)

        if success:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.update_time = datetime.now()
        else:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Broker execution failed (insufficient cash/positions)"
            order.update_time = datetime.now()

        # 4. Move to history
        if order.order_id in self._active_orders:
            del self._active_orders[order.order_id]
        self._order_history.append(order)
        self._publish_event(EventType.ORDER, order)

        # 5. Update risk manager post-trade
        if self.risk_manager is not None and success:
            self.risk_manager.post_trade_update(order)

        return order

    def _create_order(self, signal: Signal) -> Order:
        """Create an Order from a Signal."""
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        return Order(
            order_id=order_id,
            symbol=signal.symbol,
            direction=signal.direction,
            order_type=signal.order_type,
            quantity=signal.quantity,
            price=signal.price,
            status=OrderStatus.SUBMITTED,
            source=signal.reason,
        )

    def _execute_order(self, order: Order, date=None) -> bool:
        """
        Execute order via BrokerAdapter.

        In backtest mode, this calls broker.buy/sell directly (instant fill).
        In live mode, this would submit to the gateway.
        """
        exec_date = date or datetime.now()
        price = order.price or 0.0  # Market orders need price from caller

        if order.direction == Direction.BUY:
            success = self.broker.buy(exec_date, order.symbol, price, order.quantity)
        elif order.direction == Direction.SELL:
            success = self.broker.sell(exec_date, order.symbol, price, order.quantity)
        else:
            logger.error(f"Unknown direction: {order.direction}")
            return False

        if success:
            # Record trade
            order.avg_fill_price = price
            trade = self._create_trade(order, price)
            self._trade_history.append(trade)
            self._publish_event(EventType.TRADE, trade)

        return success

    def _create_trade(self, order: Order, fill_price: float) -> Trade:
        """Create a Trade record from a filled order."""
        self._trade_counter += 1
        trade_id = f"TRD-{self._trade_counter:06d}"

        commission = fill_price * order.quantity * self.broker.commission_rate

        return Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
        )

    # ──────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active (unfilled) orders, optionally filtered by symbol."""
        orders = list(self._active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all historical orders, optionally filtered by symbol."""
        orders = self._order_history
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """Get all trade records, optionally filtered by symbol."""
        trades = self._trade_history
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        return trades

    @property
    def total_orders(self) -> int:
        return len(self._order_history)

    @property
    def total_trades(self) -> int:
        return len(self._trade_history)

    @property
    def rejected_orders(self) -> List[Order]:
        return [o for o in self._order_history if o.status == OrderStatus.REJECTED]

    # ──────────────────────────────────────────
    # Event publishing
    # ──────────────────────────────────────────

    def _publish_event(self, event_type: EventType, data):
        """Publish event if EventBus is connected."""
        if self.event_bus is not None:
            self.event_bus.publish(Event(event_type=event_type, data=data))
