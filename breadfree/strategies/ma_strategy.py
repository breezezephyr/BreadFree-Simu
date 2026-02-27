"""DoubleMAStrategy — 双均线金叉/死叉策略"""

import pandas as pd

from .base_strategy import BreadFreeStrategy
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DoubleMAStrategy(BreadFreeStrategy):
    """短期均线上穿长期均线买入, 下穿卖出. 支持多标的."""

    def __init__(self, broker, short_window: int = 5, long_window: int = 20,
                 lot_size: int = 100, max_position_pct: float = 1.0, **kwargs):
        super().__init__(broker, lot_size=lot_size)
        self.short_window = short_window
        self.long_window = long_window
        self.max_position_pct = float(max_position_pct)

    def on_bar(self, date, bars: dict):
        for symbol, bar_data in bars.items():
            close_price = bar_data["close"]
            if pd.isna(close_price) or close_price <= 0:
                continue

            if symbol not in self.history:
                self.history[symbol] = []
            elif not isinstance(self.history[symbol], list):
                hist = self.history[symbol]
                if isinstance(hist, pd.DataFrame):
                    self.history[symbol] = (hist["close"] if "close" in hist.columns
                                            else hist.iloc[:, -1]).tolist()
                elif isinstance(hist, pd.Series):
                    self.history[symbol] = hist.tolist()
                else:
                    self.history[symbol] = list(hist)

            self.history[symbol].append(close_price)

            if len(self.history[symbol]) < self.long_window:
                continue

            series = pd.Series(self.history[symbol])
            short_ma = series.rolling(window=self.short_window).mean()
            long_ma = series.rolling(window=self.long_window).mean()

            # 金叉买入
            if short_ma.iloc[-2] <= long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
                if symbol not in self.broker.positions:
                    available = self.broker.cash * max(0.0, min(1.0, self.max_position_pct))
                    cost_per_share = close_price * (1 + self.broker.commission_rate)
                    if cost_per_share <= 0:
                        continue
                    qty = (int(available / cost_per_share) // self.lot_size) * self.lot_size
                    if qty == 0 and self.broker.cash >= cost_per_share * self.lot_size:
                        qty = self.lot_size
                    if qty > 0:
                        logger.info(f"[{date}] 金叉买入 {symbol}: {qty} 股 @ {close_price:.2f}")
                        self.broker.buy(date, symbol, close_price, qty)

            # 死叉卖出
            elif short_ma.iloc[-2] >= long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
                if symbol in self.broker.positions:
                    pos = self.broker.positions[symbol]
                    sell_qty = (pos.quantity // self.lot_size) * self.lot_size
                    if sell_qty == 0 and pos.quantity > 0:
                        sell_qty = pos.quantity
                    if sell_qty > 0:
                        logger.info(f"[{date}] 死叉卖出 {symbol}: {sell_qty} 股 @ {close_price:.2f}")
                        self.broker.sell(date, symbol, close_price, sell_qty)
