"""
BreadFree SimulatedGateway

A mock gateway for integration testing without connecting to a real broker.
Simulates order execution with configurable fill behavior (instant fill,
partial fill, rejection, etc.).

Usage:
    gw = SimulatedGateway()
    gw.connect()
    gw.set_prices({'510300': 5.0, '510500': 6.0})
    result = gw.submit_order(OrderRequest(...))
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime

from .base_gateway import (
    BaseGateway, GatewayStatus, OrderRequest, BarData,
)
from ..engine.models import (
    Order, Trade, Account, PositionInfo,
    Direction, OrderType, OrderStatus,
)
from ..engine.broker import Broker, Position
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimulatedGateway(BaseGateway):
    """
    Simulated gateway for testing the live trading pipeline
    without connecting to a real broker.

    Features:
    - Instant fill at given price (or last known price)
    - Simulated account with configurable initial cash
    - Position tracking
    - Supports reject simulation (for testing error handling)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(gateway_name="Simulated", config=config)
        config = config or {}

        # Internal simulated broker for state management
        initial_cash = config.get("initial_cash", 100000.0)
        commission_rate = config.get("commission_rate", 0.0003)
        self._broker = Broker(initial_cash=initial_cash, commission_rate=commission_rate)

        # Price feed (set externally or via subscribe)
        self._current_prices: Dict[str, float] = {}
        self._subscribed_symbols: List[str] = []

        # Order tracking
        self._order_counter = 0
        self._trade_counter = 0
        self._orders: Dict[str, Order] = {}

        # Simulation controls
        self._reject_next = False
        self._fill_delay_bars = 0  # 0 = instant fill

    # ── Connection ──

    def connect(self) -> bool:
        self.status = GatewayStatus.CONNECTED
        logger.info(f"[{self.gateway_name}] Connected (simulated)")
        return True

    def disconnect(self):
        self.status = GatewayStatus.DISCONNECTED
        logger.info(f"[{self.gateway_name}] Disconnected")

    # ── Market data ──

    def subscribe(self, symbols: List[str], **kwargs):
        self._subscribed_symbols = list(set(self._subscribed_symbols + symbols))
        logger.info(f"[{self.gateway_name}] Subscribed: {symbols}")

    def unsubscribe(self, symbols: List[str]):
        for s in symbols:
            if s in self._subscribed_symbols:
                self._subscribed_symbols.remove(s)

    def set_prices(self, prices: Dict[str, float]):
        """Set current market prices for simulation."""
        self._current_prices.update(prices)

    def feed_bar(self, bar: BarData):
        """Feed a bar and dispatch to callbacks."""
        self._current_prices[bar.symbol] = bar.close
        self._dispatch_bar(bar)

    # ── Order management ──

    def submit_order(self, request: OrderRequest) -> str:
        """Submit order with instant simulated fill."""
        self._order_counter += 1
        order_id = request.order_id or f"SIM-{self._order_counter:06d}"

        # Check for simulated rejection
        if self._reject_next:
            self._reject_next = False
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                direction=request.direction,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                status=OrderStatus.REJECTED,
                reject_reason="Simulated rejection",
            )
            self._orders[order_id] = order
            self._dispatch_order(order)
            return order_id

        # Determine fill price
        fill_price = request.price
        if fill_price <= 0:
            fill_price = self._current_prices.get(request.symbol, 0)
        if fill_price <= 0:
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                direction=request.direction,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                status=OrderStatus.REJECTED,
                reject_reason=f"No price available for {request.symbol}",
            )
            self._orders[order_id] = order
            self._dispatch_order(order)
            return order_id

        # Execute via internal broker
        now = datetime.now()
        if request.direction == Direction.BUY:
            success = self._broker.buy(now, request.symbol, fill_price, request.quantity)
        else:
            success = self._broker.sell(now, request.symbol, fill_price, request.quantity)

        if success:
            # Create filled order
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                direction=request.direction,
                order_type=request.order_type,
                quantity=request.quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                filled_quantity=request.quantity,
                avg_fill_price=fill_price,
                commission=fill_price * request.quantity * self._broker.commission_rate,
            )
            self._orders[order_id] = order
            self._dispatch_order(order)

            # Create trade record
            self._trade_counter += 1
            trade = Trade(
                trade_id=f"SIM-T-{self._trade_counter:06d}",
                order_id=order_id,
                symbol=request.symbol,
                direction=request.direction,
                quantity=request.quantity,
                price=fill_price,
                commission=fill_price * request.quantity * self._broker.commission_rate,
            )
            self._dispatch_trade(trade)
        else:
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                direction=request.direction,
                order_type=request.order_type,
                quantity=request.quantity,
                price=fill_price,
                status=OrderStatus.REJECTED,
                reject_reason="Insufficient cash or positions",
            )
            self._orders[order_id] = order
            self._dispatch_order(order)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (simulated: only works on pending orders)."""
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                order.update_time = datetime.now()
                self._dispatch_order(order)
                return True
        return False

    def query_orders(self) -> List[Order]:
        """Return all orders."""
        return list(self._orders.values())

    # ── Account & Position queries ──

    def query_account(self) -> Account:
        """Query simulated account state."""
        equity = self._broker.get_total_equity(self._current_prices)
        return Account(
            total_equity=equity,
            available_cash=self._broker.cash,
            frozen_cash=0.0,
            position_value=equity - self._broker.cash,
            total_pnl=equity - self._broker.initial_cash,
            initial_cash=self._broker.initial_cash,
            commission_rate=self._broker.commission_rate,
        )

    def query_positions(self) -> List[PositionInfo]:
        """Query simulated positions."""
        result = []
        for symbol, pos in self._broker.positions.items():
            market_price = self._current_prices.get(symbol, pos.avg_price)
            info = PositionInfo(
                symbol=symbol,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                market_price=market_price,
                market_value=pos.quantity * market_price,
                unrealized_pnl=(market_price - pos.avg_price) * pos.quantity,
            )
            result.append(info)
        return result

    # ── Simulation controls ──

    def simulate_reject_next(self):
        """Make the next order submission fail (for testing error handling)."""
        self._reject_next = True

    def get_broker(self) -> Broker:
        """Access internal broker (for testing)."""
        return self._broker
