"""
BreadFree FutuGateway - Hong Kong Stock Trading via Futu OpenAPI

Connects to FutuOpenD (local daemon) for:
- Real-time market data (quotes, K-lines, tickers)
- Hong Kong stock trading (place order, cancel, query)
- Account and position queries

Prerequisites:
- Install: pip install futu-api
- Run FutuOpenD locally (download from https://www.futunn.com/download/OpenAPI)
- Unlock trade with password

Configuration (config dict):
    {
        "host": "127.0.0.1",
        "port": 11111,
        "trade_password": "your_password",
        "trade_env": "SIMULATE",  # or "REAL"
        "market": "HK",           # HK, US, etc.
    }
"""

from typing import Dict, List, Optional
from datetime import datetime

from .base_gateway import (
    BaseGateway, GatewayStatus, OrderRequest, BarData, TickData,
)
from ..engine.models import (
    Order, Trade, Account, PositionInfo,
    Direction, OrderType, OrderStatus,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lazy import futu to avoid hard dependency
_futu = None


def _import_futu():
    global _futu
    if _futu is None:
        try:
            import futu
            _futu = futu
        except ImportError:
            raise ImportError(
                "futu-api is not installed. Install with: pip install futu-api\n"
                "Also ensure FutuOpenD is running locally."
            )
    return _futu


class FutuGateway(BaseGateway):
    """
    Gateway for Hong Kong stock trading via Futu OpenAPI.

    Requires FutuOpenD to be running locally as a daemon.
    Supports both simulated and real trading environments.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(gateway_name="Futu", config=config)
        config = config or {}

        self._host = config.get("host", "127.0.0.1")
        self._port = config.get("port", 11111)
        self._trade_password = config.get("trade_password", "")
        self._trade_env_str = config.get("trade_env", "SIMULATE")
        self._market = config.get("market", "HK")

        # Futu context objects (initialized on connect)
        self._quote_ctx = None
        self._trade_ctx = None

        # Order ID mapping: internal_id -> futu_order_id
        self._order_id_map: Dict[str, str] = {}

    # ── Connection ──

    def connect(self) -> bool:
        """Connect to FutuOpenD and initialize quote/trade contexts."""
        try:
            futu = _import_futu()

            # Quote context
            self._quote_ctx = futu.OpenQuoteContext(
                host=self._host, port=self._port
            )

            # Trade context (based on market)
            if self._market == "HK":
                self._trade_ctx = futu.OpenHKTradeContext(
                    host=self._host, port=self._port
                )
            elif self._market == "US":
                self._trade_ctx = futu.OpenUSTradeContext(
                    host=self._host, port=self._port
                )
            else:
                self._trade_ctx = futu.OpenHKTradeContext(
                    host=self._host, port=self._port
                )

            # Unlock trade
            if self._trade_password:
                trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                           else futu.TrdEnv.SIMULATE)
                ret, data = self._trade_ctx.unlock_trade(
                    password=self._trade_password,
                    is_unlock=True
                )
                if ret != futu.RET_OK:
                    logger.error(f"[Futu] Failed to unlock trade: {data}")
                    self.status = GatewayStatus.ERROR
                    return False

            self.status = GatewayStatus.CONNECTED
            logger.info(f"[Futu] Connected to {self._host}:{self._port} "
                       f"(market={self._market}, env={self._trade_env_str})")
            return True

        except Exception as e:
            logger.error(f"[Futu] Connection failed: {e}")
            self.status = GatewayStatus.ERROR
            return False

    def disconnect(self):
        """Disconnect and close all contexts."""
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None
        self.status = GatewayStatus.DISCONNECTED
        logger.info("[Futu] Disconnected")

    # ── Market data ──

    def subscribe(self, symbols: List[str], **kwargs):
        """Subscribe to real-time quote data."""
        if not self._quote_ctx:
            logger.error("[Futu] Not connected")
            return

        futu = _import_futu()
        sub_types = kwargs.get("sub_types", [futu.SubType.QUOTE, futu.SubType.K_DAY])

        # Add market prefix if not present
        futu_codes = [self._to_futu_code(s) for s in symbols]

        ret, data = self._quote_ctx.subscribe(futu_codes, sub_types)
        if ret == futu.RET_OK:
            logger.info(f"[Futu] Subscribed: {futu_codes}")
        else:
            logger.error(f"[Futu] Subscribe failed: {data}")

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from market data."""
        if not self._quote_ctx:
            return
        futu = _import_futu()
        futu_codes = [self._to_futu_code(s) for s in symbols]
        self._quote_ctx.unsubscribe(futu_codes, [futu.SubType.QUOTE])

    def get_snapshot(self, symbols: List[str]) -> Dict[str, dict]:
        """Get latest market snapshot for given symbols."""
        if not self._quote_ctx:
            return {}

        futu = _import_futu()
        futu_codes = [self._to_futu_code(s) for s in symbols]
        ret, data = self._quote_ctx.get_market_snapshot(futu_codes)

        if ret != futu.RET_OK:
            logger.error(f"[Futu] Snapshot failed: {data}")
            return {}

        result = {}
        for _, row in data.iterrows():
            code = self._from_futu_code(row['code'])
            result[code] = {
                'last_price': row.get('last_price', 0),
                'open': row.get('open_price', 0),
                'high': row.get('high_price', 0),
                'low': row.get('low_price', 0),
                'volume': row.get('volume', 0),
            }
        return result

    # ── Order management ──

    def submit_order(self, request: OrderRequest) -> str:
        """Submit order to Futu."""
        if not self._trade_ctx:
            logger.error("[Futu] Not connected for trading")
            return ""

        futu = _import_futu()
        trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                    else futu.TrdEnv.SIMULATE)

        # Map direction
        trd_side = futu.TrdSide.BUY if request.direction == Direction.BUY else futu.TrdSide.SELL

        # Map order type
        if request.order_type == OrderType.MARKET:
            order_type = futu.OrderType.MARKET
        else:
            order_type = futu.OrderType.NORMAL  # Limit order

        futu_code = self._to_futu_code(request.symbol)
        price = request.price if request.price > 0 else 0

        ret, data = self._trade_ctx.place_order(
            price=price,
            qty=request.quantity,
            code=futu_code,
            trd_side=trd_side,
            order_type=order_type,
            trd_env=trade_env,
            remark=f"BreadFree:{request.order_id}",
        )

        if ret == futu.RET_OK:
            if not data.empty:
                futu_order_id = str(data['order_id'].iloc[0])
                self._order_id_map[request.order_id] = futu_order_id
                logger.info(f"[Futu] Order placed: {request.symbol} "
                           f"{request.direction.value} {request.quantity} "
                           f"@ {price} -> futu_id={futu_order_id}")
                return futu_order_id
            else:
                logger.warning(f"[Futu] Order placed but response data is empty: "
                             f"{request.symbol} {request.direction.value} {request.quantity}")
                return request.order_id
        else:
            logger.error(f"[Futu] Order failed: {data}")
            return ""

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by internal or Futu order ID."""
        if not self._trade_ctx:
            return False

        futu = _import_futu()
        trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                    else futu.TrdEnv.SIMULATE)

        # Resolve Futu order ID
        futu_order_id = self._order_id_map.get(order_id, order_id)

        ret, data = self._trade_ctx.modify_order(
            futu.ModifyOrderOp.CANCEL,
            order_id=futu_order_id,
            qty=0, price=0,
            trd_env=trade_env,
        )
        if ret == futu.RET_OK:
            logger.info(f"[Futu] Order cancelled: {futu_order_id}")
            return True
        else:
            logger.error(f"[Futu] Cancel failed: {data}")
            return False

    def query_orders(self) -> List[Order]:
        """Query today's orders."""
        if not self._trade_ctx:
            return []

        futu = _import_futu()
        trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                    else futu.TrdEnv.SIMULATE)

        ret, data = self._trade_ctx.order_list_query(trd_env=trade_env)
        if ret != futu.RET_OK:
            logger.error(f"[Futu] Order query failed: {data}")
            return []

        orders = []
        for _, row in data.iterrows():
            order = Order(
                order_id=str(row.get('order_id', '')),
                symbol=self._from_futu_code(row.get('code', '')),
                direction=Direction.BUY if row.get('trd_side') == 'BUY' else Direction.SELL,
                order_type=OrderType.LIMIT,
                quantity=int(row.get('qty', 0)),
                price=float(row.get('price', 0)),
                status=self._map_order_status(row.get('order_status', '')),
                filled_quantity=int(row.get('dealt_qty', 0)),
                avg_fill_price=float(row.get('dealt_avg_price', 0)),
            )
            orders.append(order)
        return orders

    # ── Account & Position queries ──

    def query_account(self) -> Account:
        """Query Futu account info."""
        if not self._trade_ctx:
            return Account()

        futu = _import_futu()
        trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                    else futu.TrdEnv.SIMULATE)

        ret, data = self._trade_ctx.accinfo_query(trd_env=trade_env)
        if ret != futu.RET_OK:
            logger.error(f"[Futu] Account query failed: {data}")
            return Account()

        row = data.iloc[0] if not data.empty else {}
        return Account(
            total_equity=float(row.get('total_assets', 0)),
            available_cash=float(row.get('avl_withdrawal_cash', 0)),
            frozen_cash=float(row.get('frozen_cash', 0)),
            position_value=float(row.get('market_val', 0)),
            total_pnl=float(row.get('realized_pl', 0)),
        )

    def query_positions(self) -> List[PositionInfo]:
        """Query Futu positions."""
        if not self._trade_ctx:
            return []

        futu = _import_futu()
        trade_env = (futu.TrdEnv.REAL if self._trade_env_str == "REAL"
                    else futu.TrdEnv.SIMULATE)

        ret, data = self._trade_ctx.position_list_query(trd_env=trade_env)
        if ret != futu.RET_OK:
            logger.error(f"[Futu] Position query failed: {data}")
            return []

        positions = []
        for _, row in data.iterrows():
            info = PositionInfo(
                symbol=self._from_futu_code(row.get('code', '')),
                quantity=int(row.get('qty', 0)),
                avg_price=float(row.get('cost_price', 0)),
                market_price=float(row.get('market_val', 0)) / max(int(row.get('qty', 1)), 1),
                market_value=float(row.get('market_val', 0)),
                unrealized_pnl=float(row.get('pl_val', 0)),
            )
            positions.append(info)
        return positions

    # ── Helpers ──

    def _to_futu_code(self, symbol: str) -> str:
        """Convert internal symbol to Futu format (e.g., '00700' -> 'HK.00700')."""
        if '.' in symbol:
            return symbol
        return f"{self._market}.{symbol}"

    def _from_futu_code(self, futu_code: str) -> str:
        """Convert Futu format to internal symbol (e.g., 'HK.00700' -> '00700')."""
        if '.' in futu_code:
            return futu_code.split('.', 1)[1]
        return futu_code

    def _map_order_status(self, futu_status: str) -> OrderStatus:
        """Map Futu order status string to internal OrderStatus."""
        status_map = {
            'SUBMITTED': OrderStatus.SUBMITTED,
            'FILLED_ALL': OrderStatus.FILLED,
            'FILLED_PART': OrderStatus.PARTIAL_FILLED,
            'CANCELLED_ALL': OrderStatus.CANCELLED,
            'CANCELLED_PART': OrderStatus.CANCELLED,
            'FAILED': OrderStatus.REJECTED,
            'DISABLED': OrderStatus.REJECTED,
        }
        return status_map.get(futu_status, OrderStatus.PENDING)
