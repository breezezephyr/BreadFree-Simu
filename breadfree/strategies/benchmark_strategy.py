"""BenchmarkStrategy — 等权买入并持有基准策略"""

from .base_strategy import BreadFreeStrategy


class BenchmarkStrategy(BreadFreeStrategy):
    """将初始资金等权分配到所有标的, 买入后持有到结束. 用作策略对比基准."""

    def __init__(self, broker, lot_size: int = 100, **kwargs):
        super().__init__(broker, lot_size)
        self.invested: dict = {}

    def set_symbols(self, symbols: list):
        super().set_symbols(symbols)
        self.invested = {s: False for s in symbols}

    def on_bar(self, date, bars: dict):
        target_per_symbol = self.broker.initial_cash / len(self.symbols)
        cr = self.broker.commission_rate

        for symbol, bar_data in bars.items():
            if self.invested.get(symbol, False):
                continue

            price = bar_data["close"]
            max_shares = int(target_per_symbol / (price * (1 + cr)))
            quantity = (max_shares // self.lot_size) * self.lot_size

            if quantity > 0:
                cost = quantity * price * (1 + cr)
                if self.broker.cash >= cost:
                    print(f"[{date}] Benchmark Buy: Buying {quantity} shares of "
                          f"{symbol} at {price:.2f}")
                    self.broker.buy(date, symbol, price, quantity)
                    self.invested[symbol] = True
