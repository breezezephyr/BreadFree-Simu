"""
SignalEngine — 动态交易信号引擎

替代固定周期 (hold_period) 调仓, 基于多维信号动态决定:
    - 何时触发调仓 (entry/exit trigger)
    - 是否提前止盈/止损
    - 是否延长持有 (强趋势不轻易卖出)

设计原则:
    - 低频优先: 最小调仓间隔 min_hold_days, 避免高频交易
    - 信号融合: 多个信号综合评分, 不依赖单一指标
    - 自适应: 市场环境好时放宽, 差时收紧

信号类型:
    1. 动量突破信号 (Momentum Breakout)
    2. 效率分衰减信号 (Efficiency Degradation)
    3. 移动止损信号 (Trailing Stop)
    4. 成交量异动信号 (Volume Surge)
    5. 多周期动量一致性 (Multi-Period Alignment)
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    HOLD = "hold"
    REBALANCE = "rebalance"
    TRAILING_STOP = "trailing_stop"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    VOLUME_SURGE = "volume_surge"


class SignalResult:
    """单次信号评估结果"""

    __slots__ = ("trigger", "signal_type", "score", "details")

    def __init__(self, trigger: bool, signal_type: SignalType,
                 score: float = 0.0, details: str = ""):
        self.trigger = trigger
        self.signal_type = signal_type
        self.score = score
        self.details = details

    def __repr__(self):
        return (f"Signal({self.signal_type.value}, trigger={self.trigger}, "
                f"score={self.score:.2f})")


class SignalEngine:
    """
    动态交易信号引擎

    每个交易日调用 evaluate() 获取综合信号:
        - trigger=True → 应该调仓
        - trigger=False → 继续持有
    """

    def __init__(
        self,
        min_hold_days: int = 10,
        max_hold_days: int = 40,
        base_hold_days: int = 20,
        trailing_stop_pct: float = -0.08,
        efficiency_decay_threshold: float = -0.5,
        volume_surge_ratio: float = 2.0,
        momentum_reversal_threshold: float = -0.03,
        rebalance_score_threshold: float = 0.6,
    ):
        """
        Args:
            min_hold_days: 最小持仓天数 (低频保护)
            max_hold_days: 最大持仓天数 (强制触发调仓)
            base_hold_days: 基准持仓周期
            trailing_stop_pct: 移动止损比例 (如 -0.08 = 从最高点回撤 8%)
            efficiency_decay_threshold: 效率分衰减阈值
            volume_surge_ratio: 成交量异动倍数 (相对20日均量)
            momentum_reversal_threshold: 动量反转阈值
            rebalance_score_threshold: 触发调仓的综合评分阈值 (0-1)
        """
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.base_hold_days = base_hold_days
        self.trailing_stop_pct = trailing_stop_pct
        self.efficiency_decay_threshold = efficiency_decay_threshold
        self.volume_surge_ratio = volume_surge_ratio
        self.momentum_reversal_threshold = momentum_reversal_threshold
        self.rebalance_score_threshold = rebalance_score_threshold

        self._peak_equity: float = 0.0
        self._holding_peaks: Dict[str, float] = {}
        self._last_efficiency: Dict[str, float] = {}
        self._signal_history: List[dict] = []

    def reset(self):
        self._peak_equity = 0.0
        self._holding_peaks.clear()
        self._last_efficiency.clear()
        self._signal_history.clear()

    # ──────────────────────────────────────────────────────────
    # 子信号计算
    # ──────────────────────────────────────────────────────────

    def _check_trailing_stop(
        self, holdings: set, current_prices: Dict[str, float]
    ) -> SignalResult:
        """移动止损: 任一持仓从峰值回撤超过阈值 (只检查实际持仓)"""
        triggered_symbols = []
        for symbol in holdings:
            peak = self._holding_peaks.get(symbol, 0)
            if symbol not in current_prices or peak <= 0:
                continue
            current = current_prices[symbol]
            drawdown = (current / peak) - 1.0
            if drawdown < self.trailing_stop_pct:
                triggered_symbols.append(
                    f"{symbol}({drawdown:.1%})")

        if triggered_symbols:
            return SignalResult(
                trigger=True, signal_type=SignalType.TRAILING_STOP,
                score=0.9,
                details=f"移动止损触发: {', '.join(triggered_symbols)}")
        return SignalResult(trigger=False, signal_type=SignalType.TRAILING_STOP)

    def _check_efficiency_degradation(
        self, current_efficiency: Dict[str, float],
        holdings: set,
    ) -> SignalResult:
        """效率分衰减: 持仓标的效率分大幅下降"""
        degraded = []
        for symbol in holdings:
            curr = current_efficiency.get(symbol, 0)
            prev = self._last_efficiency.get(symbol)
            if prev is not None and prev > 0:
                change = curr - prev
                if change < self.efficiency_decay_threshold:
                    degraded.append(f"{symbol}({change:+.2f})")

        if degraded:
            score = min(0.8, 0.4 + 0.2 * len(degraded))
            return SignalResult(
                trigger=True,
                signal_type=SignalType.EFFICIENCY_DEGRADATION,
                score=score,
                details=f"效率衰减: {', '.join(degraded)}")
        return SignalResult(
            trigger=False, signal_type=SignalType.EFFICIENCY_DEGRADATION)

    def _check_momentum_breakout(
        self, scored_symbols: List[Tuple[str, float]],
        holdings: set,
    ) -> SignalResult:
        """
        动量突破: 池外标的效率分显著高于持仓标的 → 换仓机会

        比较 Top-N 候选 vs 当前持仓, 如果差距大则触发
        """
        if not scored_symbols or not holdings:
            return SignalResult(
                trigger=False, signal_type=SignalType.MOMENTUM_BREAKOUT)

        top_scores = [s for sym, s in scored_symbols[:5] if sym not in holdings]
        hold_scores = [s for sym, s in scored_symbols if sym in holdings]

        if not top_scores or not hold_scores:
            return SignalResult(
                trigger=False, signal_type=SignalType.MOMENTUM_BREAKOUT)

        avg_top = np.mean(top_scores)
        avg_hold = np.mean(hold_scores)

        if avg_hold <= 0 and avg_top > 0:
            return SignalResult(
                trigger=True, signal_type=SignalType.MOMENTUM_BREAKOUT,
                score=0.85,
                details=f"动量突破: 候选avg={avg_top:.2f} >> 持仓avg={avg_hold:.2f}")

        if avg_hold > 0:
            improvement = (avg_top - avg_hold) / avg_hold
        else:
            improvement = avg_top - avg_hold

        if improvement > 0.5:
            return SignalResult(
                trigger=True, signal_type=SignalType.MOMENTUM_BREAKOUT,
                score=min(0.9, 0.5 + improvement * 0.3),
                details=f"动量突破: 候选效率提升 {improvement:.0%}")

        return SignalResult(
            trigger=False, signal_type=SignalType.MOMENTUM_BREAKOUT)

    def _check_volume_surge(
        self, volume_ratios: Dict[str, float], holdings: set,
    ) -> SignalResult:
        """成交量异动: 持仓标的放量可能是趋势变化信号"""
        surges = []
        for symbol in holdings:
            ratio = volume_ratios.get(symbol, 1.0)
            if ratio >= self.volume_surge_ratio:
                surges.append(f"{symbol}({ratio:.1f}x)")

        if surges:
            return SignalResult(
                trigger=True, signal_type=SignalType.VOLUME_SURGE,
                score=0.5,
                details=f"成交量异动: {', '.join(surges)}")
        return SignalResult(
            trigger=False, signal_type=SignalType.VOLUME_SURGE)

    # ──────────────────────────────────────────────────────────
    # 综合评估
    # ──────────────────────────────────────────────────────────

    def update_peaks(self, holdings: set, current_prices: Dict[str, float],
                     equity: float):
        """更新持仓峰值 (每日调用, 只跟踪实际持仓)"""
        self._peak_equity = max(self._peak_equity, equity)
        for symbol in holdings:
            if symbol in current_prices:
                prev_peak = self._holding_peaks.get(symbol, 0)
                self._holding_peaks[symbol] = max(prev_peak, current_prices[symbol])

    def clear_holding_peak(self, symbol: str):
        """清仓时移除持仓峰值跟踪"""
        self._holding_peaks.pop(symbol, None)
        self._last_efficiency.pop(symbol, None)

    def evaluate(
        self,
        days_held: int,
        holdings: set,
        current_prices: Dict[str, float],
        current_efficiency: Dict[str, float],
        scored_symbols: List[Tuple[str, float]],
        volume_ratios: Optional[Dict[str, float]] = None,
        equity: float = 0.0,
    ) -> SignalResult:
        """
        综合评估是否应该调仓

        Args:
            days_held: 当前已持有天数
            holdings: 当前持仓标的集合
            current_prices: 当前价格
            current_efficiency: 当前各标的效率分
            scored_symbols: 全池按效率分排序的 [(symbol, score), ...]
            volume_ratios: 各标的成交量/20日均量
            equity: 当前权益

        Returns:
            SignalResult (trigger=True 表示应该调仓)
        """
        self.update_peaks(holdings, current_prices, equity)

        if days_held >= self.max_hold_days:
            result = SignalResult(
                trigger=True, signal_type=SignalType.REBALANCE,
                score=1.0,
                details=f"达到最大持仓天数 {self.max_hold_days}")
            self._record_signal(days_held, result)
            return result

        if days_held < self.min_hold_days:
            ts_signal = self._check_trailing_stop(holdings, current_prices)
            if ts_signal.trigger:
                self._record_signal(days_held, ts_signal)
                return ts_signal

            result = SignalResult(
                trigger=False, signal_type=SignalType.HOLD,
                details=f"最小持仓期保护 ({days_held}/{self.min_hold_days})")
            return result

        signals = []

        ts = self._check_trailing_stop(holdings, current_prices)
        signals.append(ts)

        ed = self._check_efficiency_degradation(current_efficiency, holdings)
        signals.append(ed)

        mb = self._check_momentum_breakout(scored_symbols, holdings)
        signals.append(mb)

        if volume_ratios:
            vs = self._check_volume_surge(volume_ratios, holdings)
            signals.append(vs)

        weights = {
            SignalType.TRAILING_STOP: 0.35,
            SignalType.EFFICIENCY_DEGRADATION: 0.25,
            SignalType.MOMENTUM_BREAKOUT: 0.25,
            SignalType.VOLUME_SURGE: 0.15,
        }

        composite_score = sum(
            s.score * weights.get(s.signal_type, 0.1)
            for s in signals if s.trigger
        )

        hold_pressure = max(0, (days_held - self.base_hold_days)) / self.max_hold_days
        composite_score += hold_pressure * 0.3

        triggered = [s for s in signals if s.trigger]
        if composite_score >= self.rebalance_score_threshold or len(triggered) >= 2:
            primary = max(triggered, key=lambda s: s.score) if triggered else signals[0]
            detail_parts = [s.details for s in triggered if s.details]
            result = SignalResult(
                trigger=True,
                signal_type=primary.signal_type,
                score=composite_score,
                details=f"综合评分 {composite_score:.2f}: " + "; ".join(detail_parts))
            self._record_signal(days_held, result)
            self._update_efficiency_history(current_efficiency)
            return result

        self._update_efficiency_history(current_efficiency)
        return SignalResult(
            trigger=False, signal_type=SignalType.HOLD, score=composite_score,
            details=f"综合评分 {composite_score:.2f} < 阈值 {self.rebalance_score_threshold}")

    def _update_efficiency_history(self, current_efficiency: Dict[str, float]):
        self._last_efficiency = dict(current_efficiency)

    def _record_signal(self, days_held: int, signal: SignalResult):
        self._signal_history.append({
            "days_held": days_held,
            "type": signal.signal_type.value,
            "score": signal.score,
            "details": signal.details,
        })

    def get_signal_summary(self) -> List[dict]:
        """获取信号历史摘要, 用于邮件报告"""
        return list(self._signal_history[-20:])

    def get_stats(self) -> dict:
        """获取信号引擎统计"""
        if not self._signal_history:
            return {"total_signals": 0}

        type_counts = {}
        for s in self._signal_history:
            t = s["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_signals": len(self._signal_history),
            "signal_types": type_counts,
            "avg_score": float(np.mean([s["score"] for s in self._signal_history])),
        }
