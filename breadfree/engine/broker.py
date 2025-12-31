class Position:
    def __init__(self, symbol, quantity, avg_price):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price

    def __repr__(self):
        return f"Position({self.symbol}, {self.quantity}, {self.avg_price:.2f})"

class Broker:
    def __init__(self, initial_cash=100000.0, commission_rate=0.0003):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # symbol -> Position
        self.commission_rate = commission_rate
        self.transaction_history = []
        self.equity_curve = []

    def buy(self, date, symbol, price, quantity):
        cost = price * quantity
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if self.cash >= total_cost:
            self.cash -= total_cost
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_quantity = pos.quantity + quantity
                # Weighted average price
                new_avg_price = (pos.quantity * pos.avg_price + cost) / new_quantity
                pos.quantity = new_quantity
                pos.avg_price = new_avg_price
            else:
                self.positions[symbol] = Position(symbol, quantity, price)
            
            self.transaction_history.append({
                'date': date,
                'action': 'BUY',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'commission': commission,
                'cash_remaining': self.cash
            })
            return True
        else:
            print(f"[{date}] Insufficient cash to buy {symbol}. Needed: {total_cost:.2f}, Available: {self.cash:.2f}")
            return False

    def sell(self, date, symbol, price, quantity):
        if symbol in self.positions and self.positions[symbol].quantity >= quantity:
            revenue = price * quantity
            commission = revenue * self.commission_rate
            net_revenue = revenue - commission
            
            self.cash += net_revenue
            pos = self.positions[symbol]
            pos.quantity -= quantity
            
            if pos.quantity == 0:
                del self.positions[symbol]
            
            self.transaction_history.append({
                'date': date,
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'commission': commission,
                'cash_remaining': self.cash
            })
            return True
        else:
            print(f"[{date}] Insufficient positions to sell {symbol}.")
            return False

    def get_total_equity(self, current_prices):
        """
        Calculate total equity (Cash + Market Value of Positions)
        :param current_prices: dict {symbol: price}
        """
        market_value = 0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.avg_price) # Use last known price if current not available
            market_value += pos.quantity * price
        return self.cash + market_value
