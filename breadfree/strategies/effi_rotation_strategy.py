"""
RotationStrategy — 多因子效率轮动策略

核心公式:
    composite_score = α * efficiency + β * momentum_accel + γ * mean_reversion_penalty

其中:
    efficiency = (momentum / period_volatility) * R²
    momentum_accel = current_momentum - lagged_momentum   (趋势加速度)
    mean_reversion_penalty = -max(0, drawdown_from_high)  (离高点越远罚分越重)

关键优化 (相比 V1):
    1. 动量加速度因子: 趋势在加强时加分, 衰减时减分
    2. 回撤惩罚: 远离近期高点的标的被降权, 避免追涨抄底
    3. 换手摩擦控制: 现有持仓享有 retention_bonus, 减少无意义换手
    4. 风险平价加权: 按波动率倒数分配仓位, 低波资产获更高权重
    5. 组合回撤熔断: 净值跌破阈值时自动减仓至现金
    6. 最低效率分门槛: 效率分 < 0 的标的不入选, 避免亏损趋势
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .base_strategy import BreadFreeStrategy
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RotationStrategy(BreadFreeStrategy):
    """多因子效率轮动策略 — 适用于 ETF / A股低频轮动场景"""

    def __init__(
        self,
        broker,
        lookback_period: int = 20,
        hold_period: int = 20,
        top_n: int = 3,
        use_efficiency: bool = True,
        lot_size: int = 100,
        min_data_ratio: float = 1.5,
        enable_risk_parity: bool = True,
        min_momentum: float = 0.0,
        accel_lookback: int = 5,
        retention_bonus: float = 0.15,
        drawdown_circuit_breaker: float = -0.15,
        **kwargs,
    ):
        """
        Args:
            broker: Broker 实例
            lookback_period: 效率分回看窗口 (交易日)
            hold_period: 调仓周期 (交易日)
            top_n: 持仓标的数量
            use_efficiency: True=效率轮动, False=纯动量轮动
            lot_size: 最小交易手数
            min_data_ratio: 最小数据量 = lookback_period * min_data_ratio
            enable_risk_parity: 启用风险平价 (波动率倒数) 加权
            min_momentum: 动量阈值 (过滤负动量标的)
            accel_lookback: 动量加速度的滞后窗口
            retention_bonus: 持仓保留加成 (降低换手)
            drawdown_circuit_breaker: 组合回撤熔断线 (如 -0.15 = 最大回撤 15% 时减仓)
        """
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.top_n = top_n
        self.use_efficiency = use_efficiency
        self.min_data_ratio = min_data_ratio
        self.min_data_length = max(int(lookback_period * min_data_ratio), 30)
        self.enable_risk_parity = enable_risk_parity
        self.min_momentum = min_momentum
        self.accel_lookback = accel_lookback
        self.retention_bonus = retention_bonus
        self.drawdown_circuit_breaker = drawdown_circuit_breaker

        self.days_counter = 0
        self.suspension_flags: Dict[str, bool] = defaultdict(bool)
        self.last_valid_prices: Dict[str, float] = {}

        self.trade_counter = 0
        self.rebalance_dates: list = []
        self.peak_equity: float = 0.0

    # ──────────────────────────────────────────────────────────────
    # 因子计算
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _linear_regression(prices: np.ndarray) -> Tuple[float, float, float]:
        """稳健线性回归, 返回 (slope, intercept, R²)"""
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
        """区间收益率 (ROC)"""
        if prices[0] <= 0:
            return 0.0
        return prices[-1] / prices[0] - 1.0

    def _calc_efficiency(self, symbol: str, window: int) -> Dict[str, float]:
        """
        计算标的的多因子得分, 返回各因子明细.

        Returns:
            dict with keys: momentum, accel, volatility, r2, efficiency,
                            drawdown_from_high, composite
            或 None (数据不足)
        """
        history = self.history[symbol]
        if len(history) < self.min_data_length:
            return None

        full_len = window + self.accel_lookback
        start_idx = max(0, len(history) - full_len)
        full_prices = np.array(history[start_idx:])
        current_prices = full_prices[-window:]

        # 1) 动量 (ROC)
        momentum = self._calc_momentum(current_prices)

        if not self.use_efficiency:
            return {
                "momentum": momentum, "accel": 0.0, "volatility": 0.0,
                "r2": 0.0, "efficiency": momentum, "drawdown_from_high": 0.0,
                "composite": momentum,
            }

        try:
            # 2) 波动率
            returns = np.diff(current_prices) / current_prices[:-1]
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            period_vol = volatility * np.sqrt(len(returns))

            # 3) 趋势质量 (R²)
            _, _, r2 = self._linear_regression(current_prices)

            # 4) 效率分 = risk-adjusted momentum * trend quality
            epsilon = 1e-6
            efficiency = (momentum / (period_vol + epsilon)) * max(r2, 0.0)

            # 5) 动量加速度
            accel = 0.0
            if len(full_prices) >= window + self.accel_lookback:
                prev_prices = full_prices[-(window + self.accel_lookback):-self.accel_lookback]
                prev_momentum = self._calc_momentum(prev_prices)
                accel = momentum - prev_momentum

            # 6) 离高点距离 (惩罚项)
            high = np.max(current_prices)
            drawdown_from_high = (current_prices[-1] / high - 1.0) if high > 0 else 0.0

            # 7) 多因子合成
            composite = efficiency + 0.3 * accel + 0.1 * drawdown_from_high

            return {
                "momentum": momentum,
                "accel": accel,
                "volatility": volatility,
                "r2": r2,
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
        """更新价格历史并处理停牌"""
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

    def _get_valid_symbols(self) -> List[str]:
        """筛选可交易标的 (排除停牌和数据不足)"""
        valid = []
        for symbol in self.symbols:
            if self.suspension_flags.get(symbol, False):
                continue
            if len(self.history.get(symbol, [])) < self.min_data_length:
                continue
            valid.append(symbol)
        return valid

    # ──────────────────────────────────────────────────────────────
    # 仓位权重计算
    # ──────────────────────────────────────────────────────────────

    def _risk_parity_weights(self, symbols: List[str], factor_data: Dict[str, dict]) -> Dict[str, float]:
        """风险平价: 权重 ∝ 1/volatility (波动率倒数)"""
        if not self.enable_risk_parity or not symbols:
            return {s: 1.0 / len(symbols) for s in symbols}

        inv_vols = {}
        for s in symbols:
            vol = factor_data.get(s, {}).get("volatility", 0.0)
            inv_vols[s] = 1.0 / (vol + 1e-6)

        total = sum(inv_vols.values())
        if total <= 0:
            return {s: 1.0 / len(symbols) for s in symbols}
        return {s: v / total for s, v in inv_vols.items()}

    # ──────────────────────────────────────────────────────────────
    # 交易执行
    # ──────────────────────────────────────────────────────────────

    def _execute_rebalance(self, date, bars: dict, target_symbols: List[str],
                           weights: Dict[str, float]):
        """执行调仓: 先卖后买, 佣金感知"""
        current_prices = {s: bars[s]["close"] for s in bars if s in self.history}
        total_equity = self.broker.get_total_equity(current_prices)
        cr = self.broker.commission_rate

        # 1) 卖出非目标仓位
        for symbol in list(self.broker.positions.keys()):
            if symbol not in target_symbols and not self.suspension_flags.get(symbol, False):
                if symbol in bars:
                    pos = self.broker.positions[symbol]
                    if pos.quantity > 0:
                        self.broker.sell(date, symbol, bars[symbol]["close"], pos.quantity)
                        logger.info(f"清仓 {symbol}: {pos.quantity} 股")
                        self.trade_counter += 1

        # 2) 买入/调整目标仓位
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

        if self.days_counter % self.hold_period != 0:
            return

        # 组合回撤熔断
        current_prices = {s: bars[s]["close"] for s in bars if s in self.history}
        equity = self.broker.get_total_equity(current_prices)
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            dd = (equity / self.peak_equity) - 1.0
            if dd < self.drawdown_circuit_breaker:
                logger.warning(f"⚠ 回撤熔断触发: {dd:.2%} < {self.drawdown_circuit_breaker:.2%}, 清仓观望")
                for symbol in list(self.broker.positions.keys()):
                    if symbol in bars:
                        pos = self.broker.positions[symbol]
                        if pos.quantity > 0:
                            self.broker.sell(date, symbol, bars[symbol]["close"], pos.quantity)
                self.rebalance_dates.append(date)
                return

        self.rebalance_dates.append(date)
        mode = "效率轮动" if self.use_efficiency else "动量轮动"
        logger.info(f"\n{'=' * 50}\n{date} 调仓日 ({mode})")
        logger.info(f"当前持仓: {list(self.broker.positions.keys())}")

        # 1) 筛选有效标的
        valid_symbols = self._get_valid_symbols()
        logger.info(f"有效标的: {len(valid_symbols)}/{len(self.symbols)}")

        # 2) 计算因子得分
        factor_data: Dict[str, dict] = {}
        for symbol in valid_symbols:
            result = self._calc_efficiency(symbol, self.lookback_period)
            if result is not None:
                factor_data[symbol] = result

        # 3) 过滤无效得分 & 负效率分
        scored = {s: d for s, d in factor_data.items() if d["composite"] > -np.inf}
        if self.use_efficiency:
            scored = {s: d for s, d in scored.items() if d["efficiency"] >= 0}

        # 4) 排序 (composite 综合得分)
        current_holdings = set(self.broker.positions.keys())
        ranking: List[Tuple[str, float]] = []
        for s, d in scored.items():
            bonus = self.retention_bonus if s in current_holdings else 0.0
            ranking.append((s, d["composite"] + bonus))

        ranking.sort(key=lambda x: x[1], reverse=True)

        # 5) 动量阈值过滤 & 选 Top-N
        if self.min_momentum > 0:
            candidates = [(s, sc) for s, sc in ranking if scored[s]["momentum"] >= self.min_momentum]
        else:
            candidates = ranking
        target_symbols = [s for s, _ in candidates[:self.top_n]]

        logger.info(f"目标持仓: {target_symbols}")

        # 6) 计算权重 & 执行
        if target_symbols:
            weights = self._risk_parity_weights(target_symbols, factor_data)
            logger.info(f"目标权重: { {s: f'{w:.2%}' for s, w in weights.items()} }")
            self._execute_rebalance(date, bars, target_symbols, weights)
        else:
            logger.info("无有效标的, 持有现金")

        equity_after = self.broker.get_total_equity(current_prices)
        logger.info(f"调仓后权益: {equity_after:.2f}")

    def get_performance_summary(self) -> dict:
        """返回策略执行摘要"""
        return {
            "total_trades": self.trade_counter,
            "rebalance_count": len(self.rebalance_dates),
            "last_rebalance": self.rebalance_dates[-1] if self.rebalance_dates else None,
        }
