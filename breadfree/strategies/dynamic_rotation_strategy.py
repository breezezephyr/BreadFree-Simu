"""
DynamicRotationStrategy — 主动发现 + 动态触发轮动策略

相比 RotationStrategy 的三大升级:
    1. 主动选股: 不局限于固定池, 每次调仓前扫描全市场寻找高效率标的
    2. 动态触发: 用 SignalEngine 替代固定 hold_period, 在最优买卖点进出场
    3. 自适应持仓: 强趋势延长持有, 弱趋势缩短持有, 止损果断执行

低频交易原则:
    - 最小持仓间隔 min_hold_days (默认 10 天), 防止高频交易
    - 成交量异动只是辅助信号, 不单独触发
    - 全市场扫描结果缓存 12 小时, 不会过度调用 API
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .base_strategy import BreadFreeStrategy
from ..engine.signal_engine import SignalEngine, SignalType
from ..data.stock_discovery import StockDiscovery
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class DynamicRotationStrategy(BreadFreeStrategy):
    """主动发现 + 动态触发轮动策略"""

    def __init__(
        self,
        broker,
        lookback_period: int = 20,
        top_n: int = 3,
        lot_size: int = 100,
        use_efficiency: bool = True,
        min_data_ratio: float = 1.5,
        enable_risk_parity: bool = False,
        min_momentum: float = 0.0,
        accel_lookback: int = 5,
        retention_bonus: float = 0.05,
        drawdown_circuit_breaker: float = -0.20,
        # 动态触发参数
        min_hold_days: int = 10,
        max_hold_days: int = 40,
        trailing_stop_pct: float = -0.08,
        rebalance_score_threshold: float = 0.6,
        # 主动发现参数
        enable_discovery: bool = True,
        max_expand: int = 10,
        discovery_min_amount: float = 5e7,
        discovery_min_circ_mv: float = 5e9,
        discovery_efficiency_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.use_efficiency = use_efficiency
        self.min_data_ratio = min_data_ratio
        self.min_data_length = max(int(lookback_period * min_data_ratio), 30)
        self.enable_risk_parity = enable_risk_parity
        self.min_momentum = min_momentum
        self.accel_lookback = accel_lookback
        self.retention_bonus = retention_bonus
        self.drawdown_circuit_breaker = drawdown_circuit_breaker
        self.enable_discovery = enable_discovery
        self.max_expand = max_expand

        self.days_counter = 0
        self.days_since_rebalance = 0
        self.suspension_flags: Dict[str, bool] = defaultdict(bool)
        self.last_valid_prices: Dict[str, float] = {}
        self.trade_counter = 0
        self.rebalance_dates: list = []
        self.peak_equity: float = 0.0

        self.signal_engine = SignalEngine(
            min_hold_days=min_hold_days,
            max_hold_days=max_hold_days,
            base_hold_days=lookback_period,
            trailing_stop_pct=trailing_stop_pct,
            rebalance_score_threshold=rebalance_score_threshold,
        )

        self._discovery: Optional[StockDiscovery] = None
        if enable_discovery:
            self._discovery = StockDiscovery(
                min_amount=discovery_min_amount,
                min_circ_mv=discovery_min_circ_mv,
                efficiency_threshold=discovery_efficiency_threshold,
                lookback_period=lookback_period,
                max_discover=max_expand * 3,
            )

        self._expanded_symbols: set = set()
        self._discovery_log: List[dict] = []

    # ──────────────────────────────────────────────────────────────
    # 因子计算 (复用 RotationStrategy 核心逻辑)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _linear_regression(prices: np.ndarray) -> Tuple[float, float, float]:
        n = len(prices)
        if n < 2:
            return 0.0, 0.0, 0.0
        try:
            x = np.arange(n)
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
            return slope, intercept, r_value ** 2
        except Exception:
            diff = prices[-1] - prices[0]
            return diff / max(n - 1, 1), prices[0], 0.0

    def _calc_momentum(self, prices: np.ndarray) -> float:
        if prices[0] <= 0:
            return 0.0
        return prices[-1] / prices[0] - 1.0

    def _calc_efficiency(self, symbol: str, window: int) -> Optional[dict]:
        history = self.history.get(symbol, [])
        if len(history) < self.min_data_length:
            return None

        full_len = window + self.accel_lookback
        start_idx = max(0, len(history) - full_len)
        full_prices = np.array(history[start_idx:])
        current_prices = full_prices[-window:]

        momentum = self._calc_momentum(current_prices)

        if not self.use_efficiency:
            return {
                "momentum": momentum, "accel": 0.0, "volatility": 0.0,
                "r2": 0.0, "efficiency": momentum, "drawdown_from_high": 0.0,
                "composite": momentum,
            }

        try:
            returns = np.diff(current_prices) / current_prices[:-1]
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            period_vol = volatility * np.sqrt(len(returns))

            _, _, r2 = self._linear_regression(current_prices)

            epsilon = 1e-6
            efficiency = (momentum / (period_vol + epsilon)) * max(r2, 0.0)

            accel = 0.0
            if len(full_prices) >= window + self.accel_lookback:
                prev_prices = full_prices[-(window + self.accel_lookback):-self.accel_lookback]
                prev_momentum = self._calc_momentum(prev_prices)
                accel = momentum - prev_momentum

            high = np.max(current_prices)
            drawdown_from_high = (current_prices[-1] / high - 1.0) if high > 0 else 0.0

            composite = efficiency + 0.3 * accel + 0.1 * drawdown_from_high

            return {
                "momentum": momentum, "accel": accel,
                "volatility": volatility, "r2": r2,
                "efficiency": efficiency,
                "drawdown_from_high": drawdown_from_high,
                "composite": composite,
            }
        except Exception as e:
            logger.error(f"因子计算异常 {symbol}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # 数据维护
    # ──────────────────────────────────────────────────────────────

    def _update_prices(self, date, bars: dict):
        for symbol in self.symbols:
            if symbol in bars:
                price = bars[symbol]["close"]
                self.history[symbol].append(price)
                self.last_valid_prices[symbol] = price
                self.suspension_flags[symbol] = False
            elif symbol in self.last_valid_prices:
                self.suspension_flags[symbol] = True
                self.history[symbol].append(self.last_valid_prices[symbol])
            else:
                if symbol not in self.history or not self.history[symbol]:
                    self.history[symbol] = []

        for symbol in self._expanded_symbols:
            if symbol not in self.symbols and symbol in bars:
                if symbol not in self.history:
                    self.history[symbol] = []
                self.history[symbol].append(bars[symbol]["close"])
                self.last_valid_prices[symbol] = bars[symbol]["close"]

    def _get_valid_symbols(self) -> List[str]:
        valid = []
        all_symbols = set(self.symbols) | self._expanded_symbols
        for symbol in all_symbols:
            if self.suspension_flags.get(symbol, False):
                continue
            if len(self.history.get(symbol, [])) < self.min_data_length:
                continue
            valid.append(symbol)
        return valid

    # ──────────────────────────────────────────────────────────────
    # 主动发现
    # ──────────────────────────────────────────────────────────────

    def _run_discovery(self, date) -> List[str]:
        """执行全市场扫描, 返回新发现的标的列表"""
        if not self._discovery:
            return []

        try:
            end_date = date.strftime("%Y%m%d") if hasattr(date, "strftime") else str(date)[:10].replace("-", "")
            expanded_pool = self._discovery.get_expanded_pool(
                end_date=end_date, max_expand=self.max_expand)

            cfg = get_config()
            fixed_pool = set(cfg.get("etf_pool", {}).keys())
            new_symbols = [s for s in expanded_pool if s not in fixed_pool]

            for sym in new_symbols:
                if sym not in self._expanded_symbols:
                    self._expanded_symbols.add(sym)
                    if sym not in self.history:
                        self.history[sym] = []
                    self._discovery_log.append({
                        "date": str(date),
                        "symbol": sym,
                        "name": expanded_pool.get(sym, sym),
                    })

            return new_symbols
        except Exception as e:
            logger.warning(f"[DynRotation] 主动发现异常: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # 权重计算
    # ──────────────────────────────────────────────────────────────

    def _calc_weights(self, symbols: List[str],
                      factor_data: Dict[str, dict]) -> Dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        equal_w = {s: 1.0 / n for s in symbols}

        if not self.enable_risk_parity:
            return equal_w

        inv_vols = {}
        for s in symbols:
            vol = factor_data.get(s, {}).get("volatility", 0.0)
            inv_vols[s] = 1.0 / max(vol, 0.005)

        inv_total = sum(inv_vols.values())
        rp_w = {s: v / inv_total for s, v in inv_vols.items()} if inv_total > 0 else equal_w

        blended = {s: 0.5 * equal_w[s] + 0.5 * rp_w[s] for s in symbols}
        max_w = 0.60
        capped = {s: min(w, max_w) for s, w in blended.items()}
        cap_total = sum(capped.values())
        if cap_total > 0:
            capped = {s: w / cap_total for s, w in capped.items()}
        return capped

    # ──────────────────────────────────────────────────────────────
    # 交易执行
    # ──────────────────────────────────────────────────────────────

    def _execute_rebalance(self, date, bars: dict, target_symbols: List[str],
                           weights: Dict[str, float]):
        current_prices = {s: bars[s]["close"] for s in bars if s in self.history}
        total_equity = self.broker.get_total_equity(current_prices)
        cr = self.broker.commission_rate

        for symbol in list(self.broker.positions.keys()):
            if symbol not in target_symbols and not self.suspension_flags.get(symbol, False):
                if symbol in bars:
                    pos = self.broker.positions[symbol]
                    if pos.quantity > 0:
                        self.broker.sell(date, symbol, bars[symbol]["close"], pos.quantity)
                        logger.info(f"清仓 {symbol}: {pos.quantity} 股")
                        self.trade_counter += 1
                        self.signal_engine.clear_holding_peak(symbol)

        for symbol in target_symbols:
            if symbol not in bars:
                continue
            price = bars[symbol]["close"]
            target_value = total_equity * weights.get(symbol, 0)
            current_value = 0.0
            if symbol in self.broker.positions:
                current_value = self.broker.positions[symbol].quantity * price

            diff = target_value - current_value

            if diff > 0:
                max_buy = self.broker.cash / (1 + cr)
                buy_value = min(diff, max_buy)
                if price > 0:
                    buy_qty = int(buy_value // price)
                    buy_qty = (buy_qty // self.lot_size) * self.lot_size
                    if buy_qty > 0:
                        self.broker.buy(date, symbol, price, buy_qty)
                        logger.info(f"买入 {symbol}: {buy_qty} 股 @ {price} "
                                    f"(目标权重 {weights[symbol]:.2%})")
                        self.trade_counter += 1
            elif diff < 0:
                if price > 0 and symbol in self.broker.positions:
                    sell_qty = int(-diff // price)
                    sell_qty = min(sell_qty, self.broker.positions[symbol].quantity)
                    sell_qty = (sell_qty // self.lot_size) * self.lot_size
                    if sell_qty > 0:
                        self.broker.sell(date, symbol, price, sell_qty)
                        logger.info(f"减仓 {symbol}: {sell_qty} 股 "
                                    f"(目标权重 {weights[symbol]:.2%})")
                        self.trade_counter += 1

    # ──────────────────────────────────────────────────────────────
    # 主逻辑
    # ──────────────────────────────────────────────────────────────

    def on_bar(self, date, bars: dict):
        self._update_prices(date, bars)
        self.days_counter += 1
        self.days_since_rebalance += 1

        current_prices = {s: bars[s]["close"] for s in bars if s in self.history}
        equity = self.broker.get_total_equity(current_prices)
        self.peak_equity = max(self.peak_equity, equity)
        current_holdings = set(self.broker.positions.keys())

        self.signal_engine.update_peaks(current_holdings, current_prices, equity)

        # 组合回撤熔断
        if self.peak_equity > 0:
            dd = (equity / self.peak_equity) - 1.0
            if dd < self.drawdown_circuit_breaker:
                logger.warning(f"⚠ 回撤熔断触发: {dd:.2%} < "
                               f"{self.drawdown_circuit_breaker:.2%}, 清仓观望")
                for symbol in list(self.broker.positions.keys()):
                    if symbol in bars:
                        pos = self.broker.positions[symbol]
                        if pos.quantity > 0:
                            self.broker.sell(date, symbol,
                                            bars[symbol]["close"], pos.quantity)
                            self.signal_engine.clear_holding_peak(symbol)
                self.rebalance_dates.append(date)
                self.days_since_rebalance = 0
                return

        # 计算当前所有标的效率分 (用于信号评估)
        valid_symbols = self._get_valid_symbols()
        factor_data: Dict[str, dict] = {}
        for symbol in valid_symbols:
            result = self._calc_efficiency(symbol, self.lookback_period)
            if result is not None:
                factor_data[symbol] = result

        current_efficiency = {s: d["efficiency"] for s, d in factor_data.items()}
        scored = {s: d for s, d in factor_data.items() if d["composite"] > -np.inf}
        if self.use_efficiency:
            scored = {s: d for s, d in scored.items() if d["efficiency"] >= 0}

        current_holdings = set(self.broker.positions.keys())
        ranking: List[Tuple[str, float]] = []
        for s, d in scored.items():
            bonus = self.retention_bonus if s in current_holdings else 0.0
            ranking.append((s, d["composite"] + bonus))
        ranking.sort(key=lambda x: x[1], reverse=True)

        # 计算成交量比率 (当日成交量 / 20日均量)
        volume_ratios = {}
        for symbol in current_holdings:
            if symbol in bars and "volume" in bars[symbol]:
                hist = self.history.get(symbol, [])
                if len(hist) > 20:
                    current_vol = float(bars[symbol].get("volume", 0))
                    avg_vol = np.mean([abs(h) for h in hist[-20:]]) if hist else 1
                    if avg_vol > 0:
                        volume_ratios[symbol] = current_vol / avg_vol

        # 信号引擎评估
        signal = self.signal_engine.evaluate(
            days_held=self.days_since_rebalance,
            holdings=current_holdings,
            current_prices=current_prices,
            current_efficiency=current_efficiency,
            scored_symbols=ranking,
            volume_ratios=volume_ratios if volume_ratios else None,
            equity=equity,
        )

        if not signal.trigger:
            return

        # 触发调仓
        self.rebalance_dates.append(date)
        self.days_since_rebalance = 0

        mode_str = f"动态触发: {signal.signal_type.value}"
        logger.info(f"\n{'=' * 50}\n{date} 调仓日 ({mode_str})")
        logger.info(f"信号详情: {signal.details}")
        logger.info(f"当前持仓: {list(current_holdings)}")
        logger.info(f"有效标的: {len(valid_symbols)}/{len(self.symbols)}")

        # 主动发现 (调仓时执行扫描)
        if self.enable_discovery and self.days_counter % 5 == 0:
            new_discoveries = self._run_discovery(date)
            if new_discoveries:
                logger.info(f"[Discovery] 新发现 {len(new_discoveries)} 个标的")

        # 选择 Top-N 目标
        if self.min_momentum > 0:
            candidates = [(s, sc) for s, sc in ranking
                          if scored.get(s, {}).get("momentum", 0) >= self.min_momentum]
        else:
            candidates = ranking
        target_symbols = [s for s, _ in candidates[:self.top_n]]

        logger.info(f"目标持仓: {target_symbols}")

        if target_symbols:
            weights = self._calc_weights(target_symbols, factor_data)
            logger.info(f"目标权重: { {s: f'{w:.2%}' for s, w in weights.items()} }")
            self._execute_rebalance(date, bars, target_symbols, weights)

            # 调仓后重置持仓峰值为当前价, 避免同一回撤反复触发止损
            if signal.signal_type == SignalType.TRAILING_STOP:
                for sym in target_symbols:
                    if sym in current_prices:
                        self.signal_engine._holding_peaks[sym] = current_prices[sym]
        else:
            logger.info("无有效标的, 持有现金")

        equity_after = self.broker.get_total_equity(current_prices)
        logger.info(f"调仓后权益: {equity_after:.2f}")

    def get_performance_summary(self) -> dict:
        return {
            "total_trades": self.trade_counter,
            "rebalance_count": len(self.rebalance_dates),
            "last_rebalance": self.rebalance_dates[-1] if self.rebalance_dates else None,
            "signal_stats": self.signal_engine.get_stats(),
            "discovery_count": len(self._discovery_log),
            "expanded_pool_size": len(self._expanded_symbols),
        }

    def get_signal_history(self) -> List[dict]:
        return self.signal_engine.get_signal_summary()

    def get_discovery_log(self) -> List[dict]:
        return list(self._discovery_log)
