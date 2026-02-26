"""
BreadFree QMTGateway - A-Share Trading via 国金 QMT/xtquant

Connects to miniQMT client (running locally on Windows) via xtquant library for:
- Real-time A-share market data
- Stock trading (buy/sell/cancel)
- Account and position queries

Prerequisites:
- Install: pip install xtquant
- Run miniQMT client locally (Windows only)
- Path typically: D:\\国金证券QMT交易端\\userdata_mini

Configuration (config dict):
    {
        "mini_qmt_path": "D:\\国金证券QMT交易端\\userdata_mini",
        "session_id": 123456,
        "account_id": "your_account_id",
    }
"""

from typing import Dict, List, Optional
from datetime import datetime

from .base_gateway import (
    BaseGateway, GatewayStatus, OrderRequest, BarData,
)
from ..engine.models import (
    Order, Trade, Account, PositionInfo,
    Direction, OrderType, OrderStatus,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid hard dependency
_xtquant_loaded = False
_xttrader = None
_xtdata = None
_xtconstant = None


def _import_xtquant():
    global _xtquant_loaded, _xttrader, _xtdata, _xtconstant
    if not _xtquant_loaded:
        try:
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
            from xtquant import xtdata, xtconstant
            _xttrader = XtQuantTrader
            _xtdata = xtdata
            _xtconstant = xtconstant
            _xtquant_loaded = True
        except ImportError:
            raise ImportError(
                "xtquant is not installed. Install with: pip install xtquant\n"
                "Also ensure miniQMT client is running locally (Windows only)."
            )
    return _xttrader, _xtdata, _xtconstant


class _QMTCallback:
    """
    Callback handler for QMT trader events.
    Routes order/trade callbacks back to the gateway.
    """

    def __init__(self, gateway: 'QMTGateway'):
        self._gateway = gateway

    def on_disconnected(self):
        logger.warning("[QMT] Disconnected from miniQMT")
        self._gateway.status = GatewayStatus.DISCONNECTED

    def on_account_status(self, status):
        logger.info(f"[QMT] Account status: {status}")

    def on_order_stock_async_response(self, response):
        """Async order response (order submitted)."""
        logger.info(f"[QMT] Order response: seq={response.seq}, "
                    f"order_id={response.order_id}")

    def on_order_callback(self, order_info):
        """Order status callback from QMT."""
        order = Order(
            order_id=str(order_info.order_id),
            symbol=order_info.stock_code,
            direction=Direction.BUY if order_info.order_type in (23, 33) else Direction.SELL,
            order_type=OrderType.LIMIT,
            quantity=order_info.order_volume,
            price=order_info.price,
            status=self._map_status(order_info.order_status),
            filled_quantity=order_info.traded_volume,
            avg_fill_price=order_info.traded_price,
        )
        self._gateway._dispatch_order(order)

    def on_trade_callback(self, trade_info):
        """Trade (fill) callback from QMT."""
        trade = Trade(
            trade_id=str(trade_info.traded_id),
            order_id=str(trade_info.order_id),
            symbol=trade_info.stock_code,
            direction=Direction.BUY if trade_info.order_type in (23, 33) else Direction.SELL,
            quantity=trade_info.traded_volume,
            price=trade_info.traded_price,
        )
        self._gateway._dispatch_trade(trade)

    def _map_status(self, qmt_status) -> OrderStatus:
        """Map QMT order status to internal OrderStatus."""
        # QMT status codes vary; common mappings:
        status_map = {
            48: OrderStatus.PENDING,      # 未报
            49: OrderStatus.SUBMITTED,    # 待报
            50: OrderStatus.SUBMITTED,    # 已报
            51: OrderStatus.SUBMITTED,    # 已报待撤
            52: OrderStatus.PARTIAL_FILLED,  # 部成
            53: OrderStatus.PARTIAL_FILLED,  # 部撤
            54: OrderStatus.FILLED,       # 已成
            55: OrderStatus.CANCELLED,    # 已撤
            56: OrderStatus.REJECTED,     # 废单
            57: OrderStatus.REJECTED,     # 已确认 (rejection confirmed)
        }
        return status_map.get(qmt_status, OrderStatus.PENDING)


class QMTGateway(BaseGateway):
    """
    Gateway for A-share trading via 国金 QMT (miniQMT + xtquant).

    The xtquant library connects from an external Python process to the
    miniQMT client running locally, providing a direct trading channel.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(gateway_name="QMT", config=config)
        config = config or {}

        self._mini_qmt_path = config.get("mini_qmt_path", "")
        self._session_id = config.get("session_id", 123456)
        self._account_id = config.get("account_id", "")

        # QMT objects (initialized on connect)
        self._trader = None
        self._callback = None

        # Order tracking
        self._order_id_map: Dict[str, int] = {}  # internal_id -> qmt_order_id

    # ── Connection ──

    def connect(self) -> bool:
        """Connect to miniQMT via xtquant."""
        try:
            XtQuantTrader, xtdata, xtconstant = _import_xtquant()

            if not self._mini_qmt_path:
                logger.error("[QMT] mini_qmt_path not configured")
                self.status = GatewayStatus.ERROR
                return False

            # Create trader instance
            self._trader = XtQuantTrader(self._mini_qmt_path, self._session_id)

            # Register callback
            self._callback = _QMTCallback(self)
            self._trader.register_callback(self._callback)

            # Start and connect
            self._trader.start()
            connect_result = self._trader.connect()

            if connect_result == 0:
                self.status = GatewayStatus.CONNECTED
                logger.info(f"[QMT] Connected to miniQMT at {self._mini_qmt_path}")

                # Subscribe account
                if self._account_id:
                    self._trader.subscribe_account(self._account_id)

                return True
            else:
                logger.error(f"[QMT] Connection failed with code: {connect_result}")
                self.status = GatewayStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"[QMT] Connection error: {e}")
            self.status = GatewayStatus.ERROR
            return False

    def disconnect(self):
        """Disconnect from miniQMT."""
        if self._trader:
            self._trader.stop()
            self._trader = None
        self.status = GatewayStatus.DISCONNECTED
        logger.info("[QMT] Disconnected")

    # ── Market data ──

    def subscribe(self, symbols: List[str], **kwargs):
        """Subscribe to real-time quotes via xtdata."""
        _, xtdata, _ = _import_xtquant()
        period = kwargs.get("period", "1d")
        for symbol in symbols:
            qmt_code = self._to_qmt_code(symbol)
            xtdata.subscribe_quote(qmt_code, period=period, count=-1)
        logger.info(f"[QMT] Subscribed: {symbols}")

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from market data."""
        _, xtdata, _ = _import_xtquant()
        for symbol in symbols:
            qmt_code = self._to_qmt_code(symbol)
            xtdata.unsubscribe_quote(qmt_code)

    def get_history_data(self, symbol: str, period: str = "1d",
                         start_time: str = "", end_time: str = "",
                         count: int = -1) -> list:
        """
        Get historical K-line data via xtdata.

        :param symbol: Stock symbol (e.g., '510300')
        :param period: '1m', '5m', '1d', etc.
        :param start_time: Start time string
        :param end_time: End time string
        :param count: Number of bars (-1 for all)
        """
        _, xtdata, _ = _import_xtquant()
        qmt_code = self._to_qmt_code(symbol)

        # Download first
        xtdata.download_history_data(qmt_code, period=period,
                                      start_time=start_time, end_time=end_time)
        # Then read local
        data = xtdata.get_local_data(
            field_list=[], stock_list=[qmt_code],
            period=period, start_time=start_time, end_time=end_time,
            count=count
        )
        return data

    # ── Order management ──

    def submit_order(self, request: OrderRequest) -> str:
        """Submit order to QMT."""
        if not self._trader:
            logger.error("[QMT] Not connected")
            return ""

        _, _, xtconstant = _import_xtquant()

        qmt_code = self._to_qmt_code(request.symbol)

        # Map direction
        if request.direction == Direction.BUY:
            order_type = xtconstant.STOCK_BUY
        else:
            order_type = xtconstant.STOCK_SELL

        # Map price type
        if request.order_type == OrderType.MARKET:
            price_type = xtconstant.LATEST_PRICE
            price = 0
        else:
            price_type = xtconstant.FIX_PRICE
            price = request.price

        # Submit
        qmt_order_id = self._trader.order_stock(
            self._account_id,
            qmt_code,
            order_type,
            request.quantity,
            price_type,
            price,
            strategy_name='BreadFree',
            order_remark=request.order_id,
        )

        if qmt_order_id and qmt_order_id > 0:
            self._order_id_map[request.order_id] = qmt_order_id
            logger.info(f"[QMT] Order placed: {request.symbol} "
                       f"{request.direction.value} {request.quantity} "
                       f"@ {price} -> qmt_id={qmt_order_id}")
            return str(qmt_order_id)
        else:
            logger.error(f"[QMT] Order failed for {request.symbol}")
            return ""

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by internal or QMT order ID."""
        if not self._trader:
            return False

        # Resolve QMT order ID
        qmt_order_id = self._order_id_map.get(order_id)
        if qmt_order_id is None:
            try:
                qmt_order_id = int(order_id)
            except ValueError:
                logger.error(f"[QMT] Unknown order ID: {order_id}")
                return False

        result = self._trader.cancel_order_stock(self._account_id, qmt_order_id)
        if result == 0:
            logger.info(f"[QMT] Order cancelled: {qmt_order_id}")
            return True
        else:
            logger.error(f"[QMT] Cancel failed: {qmt_order_id}")
            return False

    def query_orders(self) -> List[Order]:
        """Query today's orders from QMT."""
        if not self._trader:
            return []

        orders_data = self._trader.query_stock_orders(self._account_id)
        if not orders_data:
            return []

        orders = []
        for od in orders_data:
            order = Order(
                order_id=str(od.order_id),
                symbol=self._from_qmt_code(od.stock_code),
                direction=Direction.BUY if od.order_type in (23, 33) else Direction.SELL,
                order_type=OrderType.LIMIT,
                quantity=od.order_volume,
                price=od.price,
                status=self._callback._map_status(od.order_status) if self._callback else OrderStatus.PENDING,
                filled_quantity=od.traded_volume,
                avg_fill_price=od.traded_price,
            )
            orders.append(order)
        return orders

    # ── Account & Position queries ──

    def query_account(self) -> Account:
        """Query QMT account info."""
        if not self._trader:
            return Account()

        asset = self._trader.query_stock_asset(self._account_id)
        if not asset:
            return Account()

        return Account(
            total_equity=asset.total_asset,
            available_cash=asset.cash,
            frozen_cash=asset.frozen_cash,
            position_value=asset.market_value,
        )

    def query_positions(self) -> List[PositionInfo]:
        """Query QMT positions."""
        if not self._trader:
            return []

        pos_list = self._trader.query_stock_positions(self._account_id)
        if not pos_list:
            return []

        positions = []
        for p in pos_list:
            info = PositionInfo(
                symbol=self._from_qmt_code(p.stock_code),
                quantity=p.volume,
                avg_price=p.open_price,
                market_price=p.market_value / max(p.volume, 1) if p.volume > 0 else 0,
                market_value=p.market_value,
            )
            positions.append(info)
        return positions

    # ── Helpers ──

    def _to_qmt_code(self, symbol: str) -> str:
        """
        Convert internal symbol to QMT format.
        e.g., '510300' -> '510300.SH', '000001' -> '000001.SZ'
        """
        if '.' in symbol:
            return symbol

        # Determine exchange suffix
        if symbol.startswith(('5', '6', '9')):
            return f"{symbol}.SH"
        elif symbol.startswith(('0', '1', '2', '3')):
            return f"{symbol}.SZ"
        else:
            return f"{symbol}.SH"  # Default to Shanghai

    def _from_qmt_code(self, qmt_code: str) -> str:
        """Convert QMT format to internal symbol. e.g., '510300.SH' -> '510300'"""
        if '.' in qmt_code:
            return qmt_code.split('.')[0]
        return qmt_code
