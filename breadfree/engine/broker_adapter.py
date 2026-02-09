"""
BreadFree BrokerAdapter - Unified Broker Interface

Abstracts away the difference between backtest (simulated) and live trading.
Strategies interact with BrokerAdapter and don't need to know which mode they're in.

Design reference:
- Backtrader: Broker abstraction (cerebro.broker)
- vnpy: Gateway + OMS separation
- nautilus_trader: ExecutionClient interface

Implementations:
- SimulatedBrokerAdapter (broker.py) - for backtesting
- LiveBrokerAdapter (future) - wraps real gateway + OMS
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from .models import Account


class BrokerAdapter(ABC):
    """
    Abstract broker interface used by all strategies.

    Strategies call buy()/sell() on this interface. The concrete implementation
    decides how to execute: instantly (backtest) or via gateway (live).

    Attributes that strategies commonly read:
    - cash: available cash
    - positions: dict of symbol -> Position
    - commission_rate: commission rate
    - initial_cash: starting capital
    - equity_curve: historical equity records
    - closed_trades: list of closed trade records
    """

    @abstractmethod
    def buy(self, date, symbol: str, price: float, quantity: int) -> bool:
        """
        Submit a buy order.

        :param date: Current date/datetime
        :param symbol: Asset symbol
        :param price: Execution price (or limit price)
        :param quantity: Number of shares/units to buy
        :return: True if order was accepted/filled, False otherwise
        """
        ...

    @abstractmethod
    def sell(self, date, symbol: str, price: float, quantity: int) -> bool:
        """
        Submit a sell order.

        :param date: Current date/datetime
        :param symbol: Asset symbol
        :param price: Execution price (or limit price)
        :param quantity: Number of shares/units to sell
        :return: True if order was accepted/filled, False otherwise
        """
        ...

    @abstractmethod
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity (cash + position market value).

        :param current_prices: Dict of symbol -> current price
        :return: Total equity value
        """
        ...

    def get_account(self) -> Account:
        """
        Get account state snapshot.
        Default implementation builds from cash/positions.
        """
        return Account(
            total_equity=getattr(self, 'current_equity', 0.0),
            available_cash=getattr(self, 'cash', 0.0),
            initial_cash=getattr(self, 'initial_cash', 0.0),
            commission_rate=getattr(self, 'commission_rate', 0.0003),
        )

    # ──────────────────────────────────────────
    # Properties that strategies commonly access
    # ──────────────────────────────────────────
    # These are defined as abstract properties so that concrete
    # implementations MUST provide them. This ensures strategies
    # can always access broker.cash, broker.positions, etc.

    @property
    @abstractmethod
    def cash(self) -> float:
        """Available cash."""
        ...

    @property
    @abstractmethod
    def positions(self) -> dict:
        """Dict of symbol -> Position."""
        ...

    @property
    @abstractmethod
    def commission_rate(self) -> float:
        """Commission rate."""
        ...

    @property
    @abstractmethod
    def initial_cash(self) -> float:
        """Initial capital."""
        ...
