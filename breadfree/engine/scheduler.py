"""
BreadFree TradingScheduler

Manages trading session timing for live trading:
- A-share and HK market trading hours
- Trading calendar (skip weekends & holidays)
- Scheduled tasks (strategy trigger, daily settlement, nightly jobs)

Design:
- Uses the `schedule` library for time-based triggers
- Supports multiple markets with different trading hours
- Provides hooks for pre-market, trading, and post-market phases
"""

import time
import threading
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass, field

import schedule

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Market definitions
# ──────────────────────────────────────────────

class Market(Enum):
    A_SHARE = "A_SHARE"     # A股 (上海/深圳)
    HK = "HK"               # 港股
    US = "US"                # 美股


@dataclass
class TradingSession:
    """A single continuous trading session (e.g., morning session)."""
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int

    def is_in_session(self, t: datetime) -> bool:
        start = t.replace(hour=self.start_hour, minute=self.start_minute, second=0)
        end = t.replace(hour=self.end_hour, minute=self.end_minute, second=0)
        return start <= t <= end


# Market trading sessions
MARKET_SESSIONS: Dict[Market, List[TradingSession]] = {
    Market.A_SHARE: [
        TradingSession(9, 30, 11, 30),    # 上午盘
        TradingSession(13, 0, 15, 0),      # 下午盘
    ],
    Market.HK: [
        TradingSession(9, 30, 12, 0),      # 上午盘
        TradingSession(13, 0, 16, 0),      # 下午盘
    ],
    Market.US: [
        TradingSession(9, 30, 16, 0),      # 常规交易 (ET)
    ],
}


# ──────────────────────────────────────────────
# TradingCalendar
# ──────────────────────────────────────────────

class TradingCalendar:
    """
    Simple trading calendar that skips weekends.
    Can be extended with holiday lists for specific markets.
    """

    def __init__(self, market: Market = Market.A_SHARE):
        self.market = market
        self._holidays: set = set()

    def add_holidays(self, dates: List[date]):
        """Add specific holiday dates."""
        self._holidays.update(dates)

    def is_trading_day(self, d: Optional[date] = None) -> bool:
        """Check if a given date is a trading day."""
        d = d or date.today()
        # Skip weekends
        if d.weekday() >= 5:
            return False
        # Skip holidays
        if d in self._holidays:
            return False
        return True

    def next_trading_day(self, d: Optional[date] = None) -> date:
        """Get the next trading day after given date."""
        d = d or date.today()
        d += timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d


# ──────────────────────────────────────────────
# TradingScheduler
# ──────────────────────────────────────────────

class TradingScheduler:
    """
    Schedules and manages trading session tasks.

    Typical daily schedule (A-share):
        09:15  - pre_market_open: connect gateway, subscribe quotes
        09:25  - call_auction_end: collect opening data
        09:30  - market_open: trading begins
        14:50  - strategy_trigger: run daily strategy (mid-low freq)
        15:00  - market_close: trading ends
        15:05  - post_market: snapshot positions, archive logs
        21:00  - nightly_job: update data, generate reports

    Usage:
        scheduler = TradingScheduler(market=Market.A_SHARE)
        scheduler.on_pre_market(my_connect_func)
        scheduler.on_strategy_trigger(my_strategy_func)
        scheduler.on_post_market(my_settlement_func)
        scheduler.start()  # blocking
    """

    def __init__(self, market: Market = Market.A_SHARE, config: Optional[dict] = None):
        self.market = market
        self.config = config or {}
        self.calendar = TradingCalendar(market)
        self.sessions = MARKET_SESSIONS.get(market, [])

        # Configurable trigger times
        self._strategy_trigger_time = self.config.get("strategy_trigger_time", "14:50")
        self._pre_market_time = self.config.get("pre_market_time", "09:15")
        self._post_market_time = self.config.get("post_market_time", "15:05")
        self._nightly_time = self.config.get("nightly_time", "21:00")

        # Callbacks
        self._pre_market_callbacks: List[Callable] = []
        self._market_open_callbacks: List[Callable] = []
        self._strategy_trigger_callbacks: List[Callable] = []
        self._post_market_callbacks: List[Callable] = []
        self._nightly_callbacks: List[Callable] = []

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Callback registration ──

    def on_pre_market(self, callback: Callable):
        """Register callback for pre-market phase (e.g., connect gateway)."""
        self._pre_market_callbacks.append(callback)

    def on_market_open(self, callback: Callable):
        """Register callback for market open."""
        self._market_open_callbacks.append(callback)

    def on_strategy_trigger(self, callback: Callable):
        """Register callback for strategy trigger (the key time for mid-low freq)."""
        self._strategy_trigger_callbacks.append(callback)

    def on_post_market(self, callback: Callable):
        """Register callback for post-market processing."""
        self._post_market_callbacks.append(callback)

    def on_nightly(self, callback: Callable):
        """Register callback for nightly job."""
        self._nightly_callbacks.append(callback)

    # ── Schedule setup ──

    def _setup_schedule(self):
        """Configure the daily schedule using the `schedule` library."""
        schedule.clear()

        # Pre-market
        schedule.every().day.at(self._pre_market_time).do(
            self._run_if_trading_day, "pre_market", self._pre_market_callbacks
        )

        # Strategy trigger (THE key moment for mid-low freq strategies)
        schedule.every().day.at(self._strategy_trigger_time).do(
            self._run_if_trading_day, "strategy_trigger", self._strategy_trigger_callbacks
        )

        # Post-market
        schedule.every().day.at(self._post_market_time).do(
            self._run_if_trading_day, "post_market", self._post_market_callbacks
        )

        # Nightly job (runs even on non-trading days for data maintenance)
        schedule.every().day.at(self._nightly_time).do(
            self._run_callbacks, "nightly", self._nightly_callbacks
        )

        logger.info(
            f"[Scheduler] Schedule configured for {self.market.value}:\n"
            f"  Pre-market:       {self._pre_market_time}\n"
            f"  Strategy trigger:  {self._strategy_trigger_time}\n"
            f"  Post-market:       {self._post_market_time}\n"
            f"  Nightly:           {self._nightly_time}"
        )

    def _run_if_trading_day(self, phase: str, callbacks: List[Callable]):
        """Only execute callbacks if today is a trading day."""
        if not self.calendar.is_trading_day():
            logger.info(f"[Scheduler] Skipping {phase} - not a trading day")
            return
        self._run_callbacks(phase, callbacks)

    def _run_callbacks(self, phase: str, callbacks: List[Callable]):
        """Execute a list of callbacks for a given phase."""
        logger.info(f"[Scheduler] === {phase.upper()} === ({datetime.now().strftime('%H:%M:%S')})")
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"[Scheduler] {phase} callback {cb.__name__} failed: {e}")

    # ── Run control ──

    def start(self, blocking: bool = True):
        """
        Start the scheduler.

        :param blocking: If True, blocks the current thread. If False, runs in background.
        """
        self._setup_schedule()
        self._running = True
        logger.info(f"[Scheduler] Started for {self.market.value}")

        if blocking:
            self._loop()
        else:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        schedule.clear()
        logger.info("[Scheduler] Stopped")

    def _loop(self):
        """Main scheduling loop."""
        while self._running:
            schedule.run_pending()
            time.sleep(1)

    # ── Utility ──

    def is_trading_hours(self, t: Optional[datetime] = None) -> bool:
        """Check if the given time is within trading hours."""
        t = t or datetime.now()
        if not self.calendar.is_trading_day(t.date()):
            return False
        return any(session.is_in_session(t) for session in self.sessions)

    def run_now(self, phase: str = "strategy_trigger"):
        """
        Manually trigger a phase immediately (useful for testing).

        :param phase: One of 'pre_market', 'strategy_trigger', 'post_market', 'nightly'
        """
        phase_map = {
            "pre_market": self._pre_market_callbacks,
            "market_open": self._market_open_callbacks,
            "strategy_trigger": self._strategy_trigger_callbacks,
            "post_market": self._post_market_callbacks,
            "nightly": self._nightly_callbacks,
        }
        callbacks = phase_map.get(phase, [])
        if callbacks:
            self._run_callbacks(phase, callbacks)
        else:
            logger.warning(f"[Scheduler] No callbacks registered for phase: {phase}")
