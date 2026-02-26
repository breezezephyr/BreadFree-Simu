"""
BreadFree BaseGateway - Abstract Gateway Interface

Defines the contract that all broker gateways must implement.
A Gateway handles the connection to a specific broker/exchange and provides:
- Market data subscription (quotes, K-lines)
- Order submission and cancellation
- Account and position queries
- Connection lifecycle management

Design reference:
- vnpy: BaseGateway with on_event callbacks
- nautilus_trader: ExecutionClient interface

Implementations:
- SimulatedGateway: For integration testing without real broker
- FutuGateway: Hong Kong stocks via Futu OpenAPI
- QMTGateway: A-shares via 国金 QMT/xtquant
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..engine.models import (
    Order, Trade, Direction, OrderType, OrderStatus,
    Account, PositionInfo, Event, EventType,
)


# ──────────────────────────────────────────────
# Gateway status
# ──────────────────────────────────────────────

class GatewayStatus(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


# ──────────────────────────────────────────────
# Market data structures
# ──────────────────────────────────────────────

@dataclass
class BarData:
    """A single OHLCV bar."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount,
        }


@dataclass
class TickData:
    """Real-time tick data."""
    symbol: str
    last_price: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_volume: int = 0
    ask_volume: int = 0
    volume: int = 0
    timestamp: Optional[datetime] = None


# ──────────────────────────────────────────────
# Order request (gateway-level)
# ──────────────────────────────────────────────

@dataclass
class OrderRequest:
    """Order submission request sent to gateway."""
    symbol: str
    direction: Direction
    order_type: OrderType
    quantity: int
    price: float = 0.0          # 0 = market order
    order_id: str = ""          # Internal order ID (assigned by OMS)

    @classmethod
    def from_order(cls, order: Order) -> 'OrderRequest':
        """Create an OrderRequest from an Order model."""
        return cls(
            symbol=order.symbol,
            direction=order.direction,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price or 0.0,
            order_id=order.order_id,
        )


# ──────────────────────────────────────────────
# BaseGateway abstract class
# ──────────────────────────────────────────────

class BaseGateway(ABC):
    """
    Abstract base class for all broker gateways.

    A gateway encapsulates:
    1. Connection management (connect/disconnect)
    2. Market data (subscribe/unsubscribe)
    3. Order execution (submit/cancel/query)
    4. Account queries (positions/balance)

    Callbacks are used to push data back to the engine:
    - on_bar: New bar data received
    - on_tick: New tick data received
    - on_order: Order status update
    - on_trade: Trade execution report
    - on_account: Account state update
    - on_position: Position update
    """

    def __init__(self, gateway_name: str, config: Optional[dict] = None):
        self.gateway_name = gateway_name
        self.config = config or {}
        self.status = GatewayStatus.DISCONNECTED

        # Callback registrations
        self._on_bar_callbacks: List[Callable] = []
        self._on_tick_callbacks: List[Callable] = []
        self._on_order_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []
        self._on_account_callbacks: List[Callable] = []
        self._on_position_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []

    # ── Connection lifecycle ──

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.
        :return: True if connection successful
        """
        ...

    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker and clean up resources."""
        ...

    @property
    def is_connected(self) -> bool:
        return self.status == GatewayStatus.CONNECTED

    # ── Market data ──

    @abstractmethod
    def subscribe(self, symbols: List[str], **kwargs):
        """
        Subscribe to market data for the given symbols.
        :param symbols: List of symbols to subscribe
        """
        ...

    @abstractmethod
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from market data."""
        ...

    # ── Order management ──

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        """
        Submit an order to the broker.
        :param request: Order request details
        :return: Broker-assigned order ID (or internal ID)
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        :param order_id: The order ID to cancel
        :return: True if cancellation request sent successfully
        """
        ...

    @abstractmethod
    def query_orders(self) -> List[Order]:
        """Query all active/today's orders from the broker."""
        ...

    # ── Account & Position queries ──

    @abstractmethod
    def query_account(self) -> Account:
        """Query account balance and equity."""
        ...

    @abstractmethod
    def query_positions(self) -> List[PositionInfo]:
        """Query current positions."""
        ...

    # ── Callback registration ──

    def register_on_bar(self, callback: Callable):
        self._on_bar_callbacks.append(callback)

    def register_on_tick(self, callback: Callable):
        self._on_tick_callbacks.append(callback)

    def register_on_order(self, callback: Callable):
        self._on_order_callbacks.append(callback)

    def register_on_trade(self, callback: Callable):
        self._on_trade_callbacks.append(callback)

    def register_on_account(self, callback: Callable):
        self._on_account_callbacks.append(callback)

    def register_on_position(self, callback: Callable):
        self._on_position_callbacks.append(callback)

    def register_on_error(self, callback: Callable):
        self._on_error_callbacks.append(callback)

    # ── Callback dispatchers (for subclasses to call) ──

    def _dispatch_bar(self, bar: BarData):
        for cb in self._on_bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                self._dispatch_error(f"on_bar callback error: {e}")

    def _dispatch_tick(self, tick: TickData):
        for cb in self._on_tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                self._dispatch_error(f"on_tick callback error: {e}")

    def _dispatch_order(self, order: Order):
        for cb in self._on_order_callbacks:
            try:
                cb(order)
            except Exception as e:
                self._dispatch_error(f"on_order callback error: {e}")

    def _dispatch_trade(self, trade: Trade):
        for cb in self._on_trade_callbacks:
            try:
                cb(trade)
            except Exception as e:
                self._dispatch_error(f"on_trade callback error: {e}")

    def _dispatch_account(self, account: Account):
        for cb in self._on_account_callbacks:
            try:
                cb(account)
            except Exception as e:
                self._dispatch_error(f"on_account callback error: {e}")

    def _dispatch_position(self, positions: List[PositionInfo]):
        for cb in self._on_position_callbacks:
            try:
                cb(positions)
            except Exception as e:
                self._dispatch_error(f"on_position callback error: {e}")

    def _dispatch_error(self, error_msg: str):
        for cb in self._on_error_callbacks:
            try:
                cb(error_msg)
            except Exception:
                pass  # Avoid infinite loop
