"""模拟券商 — 回测环境下的订单撮合 (即时成交, 无订单簿)"""

from .broker_adapter import BrokerAdapter


class Position:
    """单只标的的持仓状态"""
    __slots__ = ("symbol", "quantity", "avg_price")

    def __init__(self, symbol: str, quantity: int, avg_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price

    def __repr__(self):
        return f"Position({self.symbol}, qty={self.quantity}, avg={self.avg_price:.2f})"


class Broker(BrokerAdapter):
    """
    回测专用模拟券商.

    - 实现 BrokerAdapter 接口
    - 订单按给定价格即时成交 (无滑点、无深度模拟)
    - 使用加权平均成本法计算持仓均价
    - 双边收取佣金, 自动记录交易流水和已平仓盈亏
    """

    def __init__(self, initial_cash: float = 100000.0, commission_rate: float = 0.0003):
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._positions: dict = {}
        self._commission_rate = commission_rate
        self.transaction_history: list = []
        self.equity_curve: list = []
        self.current_equity = initial_cash
        self.closed_trades: list = []

    # ── BrokerAdapter 属性 ──

    @property
    def cash(self) -> float:
        return self._cash

    @cash.setter
    def cash(self, value: float):
        self._cash = value

    @property
    def positions(self) -> dict:
        return self._positions

    @positions.setter
    def positions(self, value: dict):
        self._positions = value

    @property
    def commission_rate(self) -> float:
        return self._commission_rate

    @commission_rate.setter
    def commission_rate(self, value: float):
        self._commission_rate = value

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    @initial_cash.setter
    def initial_cash(self, value: float):
        self._initial_cash = value

    # ── 交易方法 ──

    def buy(self, date, symbol: str, price: float, quantity: int) -> bool:
        cost = price * quantity
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if self.cash < total_cost:
            print(f"[{date}] 现金不足: {symbol} 需 {total_cost:.2f}, 可用 {self.cash:.2f}")
            return False

        self.cash -= total_cost
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_qty = pos.quantity + quantity
            pos.avg_price = (pos.quantity * pos.avg_price + cost) / new_qty
            pos.quantity = new_qty
        else:
            self.positions[symbol] = Position(symbol, quantity, price)

        self.transaction_history.append({
            "date": date, "action": "BUY", "symbol": symbol,
            "price": price, "quantity": quantity,
            "commission": commission, "cash_remaining": self.cash,
        })
        self.current_equity = self.get_total_equity({symbol: price})
        return True

    def sell(self, date, symbol: str, price: float, quantity: int) -> bool:
        if symbol not in self.positions or self.positions[symbol].quantity < quantity:
            print(f"[{date}] 仓位不足: {symbol}")
            return False

        revenue = price * quantity
        commission = revenue * self.commission_rate
        net_revenue = revenue - commission

        pos = self.positions[symbol]
        trade_return = (price - pos.avg_price) / pos.avg_price if pos.avg_price > 0 else 0.0
        pnl = (price - pos.avg_price) * quantity - commission

        self.closed_trades.append({
            "symbol": symbol, "sell_date": date,
            "buy_price": pos.avg_price, "sell_price": price,
            "quantity": quantity, "pnl": pnl,
            "return_pct": trade_return,
        })

        self.cash += net_revenue
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]

        self.transaction_history.append({
            "date": date, "action": "SELL", "symbol": symbol,
            "price": price, "quantity": quantity,
            "commission": commission, "cash_remaining": self.cash,
        })
        self.current_equity = self.get_total_equity({symbol: price})
        return True

    def get_total_equity(self, current_prices: dict) -> float:
        """总权益 = 现金 + 持仓市值"""
        market_value = sum(
            pos.quantity * current_prices.get(sym, pos.avg_price)
            for sym, pos in self.positions.items()
        )
        return self.cash + market_value
