"""
BreadFree LiveEngine - Real-time Trading Engine

The live counterpart to BacktestEngine. Connects to a real broker gateway,
feeds real-time data to strategies, and executes trades.

Key differences from BacktestEngine:
- BacktestEngine: loops through historical dates calling on_bar()
- LiveEngine: listens to real-time data, triggers on_bar() at scheduled times

Workflow:
    1. Connect to gateway (Futu / QMT)
    2. Subscribe market data for configured symbols
    3. At scheduled time (e.g., 14:50), collect latest bars and run strategy
    4. Strategy generates signals via broker interface
    5. Orders go through RiskManager -> Gateway for execution
    6. Post-market: snapshot positions, archive logs

Usage:
    engine = LiveEngine(config)
    engine.run()  # blocking
"""

import os
from datetime import datetime, date
from typing import Optional, Dict, List

import pandas as pd

from .broker import Broker
from .broker_adapter import BrokerAdapter
from .event_bus import EventBus
from .order_manager import OrderManager
from .risk_manager import RiskManager
from .scheduler import TradingScheduler, Market
from .models import Event, EventType, Order, Trade

from ..gateway.base_gateway import BaseGateway, BarData, GatewayStatus
from ..data.data_fetcher import DataFetcher
from ..data.database import get_db_manager
from ..data.live_store import LiveTradeStore
from ..monitor.alert_manager import AlertManager, AlertLevel
from ..monitor.audit_logger import AuditLogger
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LiveEngine:
    """
    Real-time trading engine for live/paper trading.

    Orchestrates:
    - Gateway connection and market data
    - Strategy execution at scheduled times
    - Order management with risk controls
    - Position and account tracking
    - State persistence and recovery
    """

    def __init__(self, config: dict):
        """
        Initialize LiveEngine from configuration.

        :param config: Configuration dict (typically loaded from config/live.yaml)

        Expected config structure:
        {
            "gateway": {"type": "futu", "host": ..., "port": ..., ...},
            "market": "A_SHARE",
            "symbols": ["510300", "510500", ...],
            "strategy": {"name": "RotationStrategy", "params": {...}},
            "risk": {"max_position_pct": 0.4, ...},
            "scheduler": {"strategy_trigger_time": "14:50", ...},
            "initial_cash": 100000,
        }
        """
        self.config = config

        # ── Core components ──
        self.event_bus = EventBus()
        self.gateway: Optional[BaseGateway] = None
        self.broker: Optional[BrokerAdapter] = None
        self.strategy = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.scheduler: Optional[TradingScheduler] = None

        # ── Data ──
        self.symbols: List[str] = config.get("symbols", [])
        self.data_fetcher = DataFetcher(
            data_dir="breadfree/data/cache",
            data_source=config.get("data_source", "akshare"),
        )

        # ── Persistence ──
        db_path = config.get("live_db_path", "live_trading.db")
        self.store = LiveTradeStore(db_path)

        # ── Monitoring ──
        alert_config = config.get("alert", {})
        self.alert_manager = AlertManager(alert_config)
        self.audit = AuditLogger(db_path)

        # ── State ──
        self._latest_bars: Dict[str, dict] = {}
        self._is_running = False
        self._today: Optional[date] = None

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all engine components from config."""
        # 1. Gateway
        self.gateway = self._create_gateway()

        # 2. Broker (uses gateway's internal broker for simulated,
        #    or wraps gateway for live)
        self.broker = self._create_broker()

        # 3. Risk Manager
        risk_config = self.config.get("risk", {})
        self.risk_manager = RiskManager(risk_config)

        # 4. Order Manager
        self.order_manager = OrderManager(
            broker=self.broker,
            risk_manager=self.risk_manager,
            event_bus=self.event_bus,
        )

        # 5. Strategy
        self.strategy = self._create_strategy()

        # 6. Scheduler
        market_str = self.config.get("market", "A_SHARE")
        market = Market[market_str] if market_str in Market.__members__ else Market.A_SHARE
        scheduler_config = self.config.get("scheduler", {})
        self.scheduler = TradingScheduler(market=market, config=scheduler_config)

        # Register scheduler callbacks
        self.scheduler.on_pre_market(self._on_pre_market)
        self.scheduler.on_strategy_trigger(self._on_strategy_trigger)
        self.scheduler.on_post_market(self._on_post_market)
        self.scheduler.on_nightly(self._on_nightly)

        # Register gateway callbacks
        if self.gateway:
            self.gateway.register_on_bar(self._on_bar_received)
            self.gateway.register_on_error(self._on_gateway_error)

        # Register event bus listeners for persistence + audit
        self.event_bus.subscribe(EventType.ORDER, self._persist_order)
        self.event_bus.subscribe(EventType.TRADE, self._persist_trade)
        self.event_bus.subscribe(EventType.ORDER, self._audit_order)
        self.event_bus.subscribe(EventType.TRADE, self._audit_trade)

    def _create_gateway(self) -> BaseGateway:
        """Create gateway based on config."""
        gw_config = self.config.get("gateway", {})
        gw_type = gw_config.get("type", "simulated").lower()

        if gw_type == "futu":
            from ..gateway.futu_gateway import FutuGateway
            return FutuGateway(gw_config)
        elif gw_type == "qmt":
            from ..gateway.qmt_gateway import QMTGateway
            return QMTGateway(gw_config)
        else:
            # Default to simulated
            from ..gateway.simulated_gateway import SimulatedGateway
            return SimulatedGateway(gw_config)

    def _create_broker(self) -> BrokerAdapter:
        """Create broker adapter."""
        # For simulated gateway, use its internal broker
        from ..gateway.simulated_gateway import SimulatedGateway
        if isinstance(self.gateway, SimulatedGateway):
            return self.gateway.get_broker()

        # For live gateways, use a Broker that syncs with gateway state
        initial_cash = self.config.get("initial_cash", 100000.0)
        commission_rate = self.config.get("commission_rate", 0.0003)
        return Broker(initial_cash=initial_cash, commission_rate=commission_rate)

    def _create_strategy(self):
        """Create strategy instance from config."""
        strategy_config = self.config.get("strategy", {})
        strategy_name = strategy_config.get("name", "RotationStrategy")
        strategy_params = strategy_config.get("params", {})

        # Import strategy classes (lazy import to avoid circular dependency)
        from ..strategies.effi_rotation_strategy import RotationStrategy
        from ..strategies.effi_agent_strategy import EffiAgentRotationStrategy
        from ..strategies.ma_strategy import DoubleMAStrategy
        from ..strategies.benchmark_strategy import BenchmarkStrategy
        from ..strategies.triple_momentum_strategy import TripleMomentumStrategy

        strategy_map = {
            "RotationStrategy": RotationStrategy,
            "EffiA": EffiAgentRotationStrategy,
            "DoubleMAStrategy": DoubleMAStrategy,
            "BenchmarkStrategy": BenchmarkStrategy,
            "TripleMomentumStrategy": TripleMomentumStrategy,
        }

        strategy_cls = strategy_map.get(strategy_name, RotationStrategy)
        lot_size = self.config.get("lot_size", 100)

        strategy = strategy_cls(self.broker, lot_size=lot_size, **strategy_params)
        strategy.set_symbols(self.symbols)

        logger.info(f"[LiveEngine] Strategy: {strategy_name} with params: {strategy_params}")
        return strategy

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def run(self):
        """
        Start the live engine (blocking).

        This connects to the gateway, subscribes to market data,
        and starts the scheduler loop.
        """
        logger.info("[LiveEngine] Starting...")
        self._is_running = True
        self.audit.log_system_event("START", f"LiveEngine starting: "
                                    f"gateway={self.config.get('gateway', {}).get('type', 'simulated')}, "
                                    f"strategy={self.config.get('strategy', {}).get('name', '?')}")

        # Connect gateway
        if self.gateway and not self.gateway.is_connected:
            if not self.gateway.connect():
                logger.error("[LiveEngine] Gateway connection failed. Aborting.")
                self.alert_manager.send_critical("Gateway connection failed at startup!")
                self.audit.log_gateway_event(
                    self.gateway.gateway_name, "CONNECT_FAIL",
                    "Connection failed at startup", level="ERROR")
                return

        # Subscribe market data
        if self.gateway and self.symbols:
            self.gateway.subscribe(self.symbols)

        # Preload history for strategy warmup
        self._preload_history()

        logger.info(f"[LiveEngine] Running with {len(self.symbols)} symbols")
        logger.info(f"[LiveEngine] Gateway: {self.gateway.gateway_name if self.gateway else 'None'}")

        # Start scheduler (blocking)
        try:
            self.scheduler.start(blocking=True)
        except KeyboardInterrupt:
            logger.info("[LiveEngine] Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Gracefully stop the engine."""
        logger.info("[LiveEngine] Stopping...")
        self._is_running = False
        self.audit.log_system_event("STOP", "LiveEngine stopping")

        if self.scheduler:
            self.scheduler.stop()

        if self.gateway:
            self.gateway.disconnect()

        logger.info("[LiveEngine] Stopped")

    # ──────────────────────────────────────────
    # Scheduler callbacks
    # ──────────────────────────────────────────

    def _on_pre_market(self):
        """Pre-market: reconnect gateway, refresh data."""
        logger.info("[LiveEngine] Pre-market phase")
        self._today = date.today()

        # Reconnect if needed
        if self.gateway and not self.gateway.is_connected:
            logger.info("[LiveEngine] Reconnecting gateway...")
            self.gateway.connect()
            if self.symbols:
                self.gateway.subscribe(self.symbols)

        # Reset daily risk counters
        if self.risk_manager:
            self.risk_manager.reset_daily()

    def _on_strategy_trigger(self):
        """
        Strategy trigger: THE core moment for mid-low frequency strategies.

        Collects latest market data and calls strategy.on_bar().
        For daily strategies, this typically runs at 14:50.
        """
        logger.info("[LiveEngine] Strategy trigger phase")

        # Collect latest bars
        bars = self._collect_latest_bars()
        if not bars:
            logger.warning("[LiveEngine] No bar data available, skipping strategy trigger")
            return

        # Run strategy
        now = datetime.now()
        try:
            logger.info(f"[LiveEngine] Running strategy with {len(bars)} symbols...")
            self.strategy.on_bar(now, bars)
            logger.info("[LiveEngine] Strategy execution completed")
            self.audit.log_strategy_decision(
                self.strategy.__class__.__name__,
                signals=[],
                reasoning=f"Triggered at {now.strftime('%H:%M:%S')} with {len(bars)} symbols",
            )
        except Exception as e:
            logger.error(f"[LiveEngine] Strategy execution failed: {e}", exc_info=True)
            self.alert_manager.send_critical(f"Strategy execution failed: {e}")
            self.audit.log_system_event("STRATEGY_ERROR", str(e), level="ERROR")

        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.LOG,
            data={"phase": "strategy_trigger", "bars_count": len(bars)},
        ))

    def _on_post_market(self):
        """Post-market: snapshot positions, persist state, calculate metrics."""
        logger.info("[LiveEngine] Post-market phase")
        today = date.today()

        # Query final account and position state
        account = None
        positions = []
        if self.gateway and self.gateway.is_connected:
            account = self.gateway.query_account()
            positions = self.gateway.query_positions()
            logger.info(f"[LiveEngine] EOD Account: equity={account.total_equity:.2f}, "
                       f"cash={account.available_cash:.2f}")
            logger.info(f"[LiveEngine] EOD Positions: {len(positions)} holdings")

            # Update equity tracking for risk manager
            if self.risk_manager:
                self.risk_manager.update_equity(account.total_equity)

        # Persist position snapshot
        if positions:
            try:
                self.store.save_position_snapshot(today, positions)
                logger.info(f"[LiveEngine] Saved position snapshot: {len(positions)} positions")
            except Exception as e:
                logger.error(f"[LiveEngine] Failed to save position snapshot: {e}")

        # Persist equity curve point
        if account:
            try:
                self.store.save_equity_point(
                    record_date=today,
                    total_equity=account.total_equity,
                    available_cash=account.available_cash,
                    position_value=account.position_value,
                    total_pnl=account.total_pnl,
                )
                logger.info(f"[LiveEngine] Saved equity point: {account.total_equity:.2f}")
            except Exception as e:
                logger.error(f"[LiveEngine] Failed to save equity point: {e}")

        # Save engine state for recovery
        try:
            self.store.save_state("last_post_market", today.isoformat())
            self.store.save_state("last_equity", str(account.total_equity) if account else "0")
            self.store.save_state("strategy", self.strategy.__class__.__name__ if self.strategy else "")
        except Exception as e:
            logger.error(f"[LiveEngine] Failed to save engine state: {e}")

        # Send daily summary alert
        try:
            account_info = {
                "total_equity": account.total_equity if account else 0,
                "available_cash": account.available_cash if account else 0,
                "total_pnl": account.total_pnl if account else 0,
            }
            today_trades = self.store.get_today_trades(today)
            risk_stats = self.risk_manager.get_stats() if self.risk_manager else {}
            self.alert_manager.send_daily_summary(account_info, today_trades, risk_stats)
        except Exception as e:
            logger.error(f"[LiveEngine] Failed to send daily summary: {e}")

        self.audit.log_system_event("POST_MARKET", f"EOD completed: "
                                    f"equity={account.total_equity:.2f}" if account else "no account data")

    def _on_nightly(self):
        """Nightly job: update data, generate reports."""
        logger.info("[LiveEngine] Nightly job phase")
        # Future: data update, report generation, LLM review

    # ──────────────────────────────────────────
    # Persistence callbacks
    # ──────────────────────────────────────────

    def _persist_order(self, event: Event):
        """Persist order to database when order event is received."""
        try:
            order = event.data
            if isinstance(order, Order):
                self.store.save_order(order)
        except Exception as e:
            logger.error(f"[LiveEngine] Failed to persist order: {e}")

    def _persist_trade(self, event: Event):
        """Persist trade to database when trade event is received."""
        try:
            trade = event.data
            if isinstance(trade, Trade):
                self.store.save_trade(trade)
        except Exception as e:
            logger.error(f"[LiveEngine] Failed to persist trade: {e}")

    # ──────────────────────────────────────────
    # Audit callbacks
    # ──────────────────────────────────────────

    def _audit_order(self, event: Event):
        """Write order event to audit log + send alerts for rejections."""
        try:
            order = event.data
            if isinstance(order, Order):
                self.audit.log_order(order)
                # Alert on rejected orders
                if order.status.value == "REJECTED":
                    self.alert_manager.send_order_alert({
                        "status": order.status.value,
                        "symbol": order.symbol,
                        "direction": order.direction.value,
                        "quantity": order.quantity,
                        "reject_reason": order.reject_reason,
                    })
        except Exception as e:
            logger.error(f"[LiveEngine] Audit order failed: {e}")

    def _audit_trade(self, event: Event):
        """Write trade event to audit log."""
        try:
            trade = event.data
            if isinstance(trade, Trade):
                self.audit.log_trade(trade)
        except Exception as e:
            logger.error(f"[LiveEngine] Audit trade failed: {e}")

    # ──────────────────────────────────────────
    # Data handling
    # ──────────────────────────────────────────

    def _on_bar_received(self, bar: BarData):
        """Callback when gateway pushes a new bar."""
        self._latest_bars[bar.symbol] = bar.to_dict()

    def _on_gateway_error(self, error_msg: str):
        """Callback when gateway reports an error."""
        logger.error(f"[LiveEngine] Gateway error: {error_msg}")
        self.alert_manager.send_warning(f"Gateway error: {error_msg}")
        self.audit.log_gateway_event(
            self.gateway.gateway_name if self.gateway else "unknown",
            "ERROR", error_msg, level="ERROR")

    def _collect_latest_bars(self) -> Dict[str, dict]:
        """
        Collect latest bar data for all subscribed symbols.

        For mid-low frequency (daily) strategies, this fetches today's
        OHLCV data. Falls back to data_fetcher if gateway doesn't have it.
        """
        bars = {}

        # Try gateway snapshot first
        if self.gateway and self.gateway.is_connected and hasattr(self.gateway, 'get_snapshot'):
            try:
                snapshot = self.gateway.get_snapshot(self.symbols)
                if snapshot:
                    for symbol, data in snapshot.items():
                        bars[symbol] = data
                        # Also update strategy history
                        if hasattr(self.strategy, 'history') and symbol in self.strategy.history:
                            close = data.get('last_price') or data.get('close', 0)
                            if close > 0:
                                self.strategy.history[symbol].append(close)
                    return bars
            except Exception as e:
                logger.warning(f"[LiveEngine] Snapshot failed: {e}")

        # Fallback: use cached latest bars from callbacks
        if self._latest_bars:
            for symbol, bar_data in self._latest_bars.items():
                bars[symbol] = bar_data
                # Update strategy history
                if hasattr(self.strategy, 'history') and symbol in self.strategy.history:
                    close = bar_data.get('close', 0)
                    if close > 0:
                        self.strategy.history[symbol].append(close)
            return bars

        # Last resort: fetch from data source
        logger.info("[LiveEngine] Fetching latest data from data source...")
        today_str = date.today().strftime("%Y%m%d")
        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_a_stock_daily(symbol, today_str, today_str)
                if not df.empty:
                    row = df.iloc[-1]
                    bar = {
                        'open': row.get('open', 0),
                        'high': row.get('high', 0),
                        'low': row.get('low', 0),
                        'close': row.get('close', 0),
                        'volume': row.get('volume', 0),
                    }
                    bars[symbol] = bar
                    # Update strategy history
                    if hasattr(self.strategy, 'history') and symbol in self.strategy.history:
                        if bar['close'] > 0:
                            self.strategy.history[symbol].append(bar['close'])
            except Exception as e:
                logger.warning(f"[LiveEngine] Failed to fetch {symbol}: {e}")

        return bars

    def _preload_history(self):
        """Preload historical data for strategy warmup."""
        from datetime import timedelta

        logger.info("[LiveEngine] Preloading historical data for strategy warmup...")
        db_manager = get_db_manager()

        end_date = date.today().strftime("%Y%m%d")
        start_dt = date.today() - timedelta(days=90)
        start_date = start_dt.strftime("%Y%m%d")

        warmup_data = {}
        for symbol in self.symbols:
            df = db_manager.get_daily_data(symbol, start_date, end_date)
            if df.empty:
                df = self.data_fetcher.fetch_a_stock_daily(symbol, start_date, end_date)
            if not df.empty:
                warmup_data[symbol] = df

        if warmup_data and hasattr(self.strategy, 'preload_history'):
            self.strategy.preload_history(warmup_data)
            logger.info(f"[LiveEngine] Preloaded history for {len(warmup_data)} symbols")

    # ──────────────────────────────────────────
    # Manual triggers (for testing / debugging)
    # ──────────────────────────────────────────

    def trigger_strategy_now(self):
        """Manually trigger strategy execution (for testing)."""
        self._on_strategy_trigger()

    def get_status(self) -> dict:
        """Get current engine status."""
        store_summary = {}
        try:
            store_summary = self.store.get_summary()
        except Exception:
            pass

        return {
            "is_running": self._is_running,
            "gateway": {
                "name": self.gateway.gateway_name if self.gateway else None,
                "status": self.gateway.status.value if self.gateway else "N/A",
            },
            "strategy": self.strategy.__class__.__name__ if self.strategy else None,
            "symbols": self.symbols,
            "symbols_with_data": len(self._latest_bars),
            "risk_stats": self.risk_manager.get_stats() if self.risk_manager else {},
            "oms_stats": {
                "total_orders": self.order_manager.total_orders if self.order_manager else 0,
                "total_trades": self.order_manager.total_trades if self.order_manager else 0,
            },
            "persistence": store_summary,
        }
