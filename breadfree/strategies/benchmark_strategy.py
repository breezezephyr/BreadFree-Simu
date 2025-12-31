from .base_strategy import Strategy

class BenchmarkStrategy(Strategy):
    """
    Market Benchmark Strategy (Buy and Hold)
    Simply buys the asset on the first available opportunity and holds it until the end.
    Used as a baseline to compare other strategies against the market performance.
    """
    def __init__(self, broker, lot_size=100):
        super().__init__(broker, lot_size)
        self.symbol = None
        self.invested = False

    def set_symbol(self, symbol):
        self.symbol = symbol

    def on_bar(self, date, bar_data):
        # If already invested, do nothing (Hold)
        if self.invested:
            return

        # Buy on the first day (or first opportunity)
        close_price = bar_data['close']
        available_cash = self.broker.cash
        
        # Calculate max quantity
        # Estimate cost including commission (approximate)
        # We leave a small buffer for commission to avoid rejection
        max_shares = int(available_cash / (close_price * (1 + self.broker.commission_rate)))
        
        # Round down to nearest lot_size
        quantity = (max_shares // self.lot_size) * self.lot_size
        
        if quantity > 0:
            print(f"[{date}] Benchmark Buy: Buying {quantity} shares of {self.symbol} at {close_price:.2f}")
            self.broker.buy(date, self.symbol, close_price, quantity)
            self.invested = True
        else:
            print(f"[{date}] Insufficient cash to buy benchmark position. Cash: {available_cash:.2f}, Price: {close_price:.2f}")
