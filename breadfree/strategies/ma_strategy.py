from .base_strategy import Strategy
import pandas as pd

class DoubleMAStrategy(Strategy):
    def __init__(self, broker, short_window=5, long_window=20, lot_size=100):
        super().__init__(broker, lot_size=lot_size)
        self.short_window = short_window
        self.long_window = long_window
        self.history = [] # Keep track of close prices
        self.symbol = None

    def set_symbol(self, symbol):
        self.symbol = symbol

    def preload_history(self, history_df):
        if not history_df.empty:
            self.history = history_df['close'].tolist()
            print(f"DoubleMAStrategy: Preloaded {len(self.history)} days of history.")

    def on_bar(self, date, bar_data):
        # bar_data is expected to be a row from the dataframe
        close_price = bar_data['close']
        self.history.append(close_price)

        if len(self.history) < self.long_window:
            return

        # Calculate MAs
        short_ma = pd.Series(self.history).rolling(window=self.short_window).mean().iloc[-1]
        long_ma = pd.Series(self.history).rolling(window=self.long_window).mean().iloc[-1]
        
        # Previous MAs to check for crossover
        prev_short_ma = pd.Series(self.history).rolling(window=self.short_window).mean().iloc[-2]
        prev_long_ma = pd.Series(self.history).rolling(window=self.long_window).mean().iloc[-2]

        # Check for crossover
        # Golden Cross (Short crosses above Long) -> Buy
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            # Buy signal
            # Simple logic: Buy as many shares as possible (in lots of lot_size)
            if self.symbol not in self.broker.positions:
                print(f"[{date}] Golden Cross: Buying {self.symbol}")
                # Calculate max quantity
                available_cash = self.broker.cash
                # Estimate cost including commission (approximate)
                max_shares = int(available_cash / (close_price * (1 + self.broker.commission_rate)))
                # Round down to nearest lot_size
                quantity = (max_shares // self.lot_size) * self.lot_size
                
                if quantity > 0:
                    self.broker.buy(date, self.symbol, close_price, quantity)
                else:
                    print(f"[{date}] Insufficient cash to buy {self.lot_size} shares of {self.symbol}. Cash: {available_cash:.2f}, Price: {close_price:.2f}")

        # Death Cross (Short crosses below Long) -> Sell
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            # Sell signal
            if self.symbol in self.broker.positions:
                print(f"[{date}] Death Cross: Selling {self.symbol}")
                pos = self.broker.positions[self.symbol]
                self.broker.sell(date, self.symbol, close_price, pos.quantity)

class 