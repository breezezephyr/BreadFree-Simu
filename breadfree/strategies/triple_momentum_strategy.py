"""TripleMomentumStrategy — 三因子动量轮动策略

结合乖离动量、斜率动量、效率动量三个因子进行标的轮动.
每个因子 Z-Score 标准化后等权合成, 选择综合得分最高的标的全仓持有.
支持调仓阈值 (只有新标的显著优于旧标的时才切换, 降低换手).
"""

import numpy as np
import pandas as pd
from scipy import stats

from .base_strategy import BreadFreeStrategy
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TripleMomentumStrategy(BreadFreeStrategy):

    def __init__(self, broker, bias_n: int = 24, momentum_day: int = 25,
                 slope_n: int = 20, hold_period: int = 20, lot_size: int = 100,
                 rebalance_threshold: float = 1.5, **kwargs):
        """
        Args:
            bias_n: 乖离率均线窗口
            momentum_day: 乖离率回归窗口
            slope_n: 斜率/效率因子窗口
            hold_period: 调仓周期 (天)
            rebalance_threshold: 新标的得分 > 旧标的 * 该倍数 时才切换
        """
        super().__init__(broker, lot_size=lot_size)
        self.bias_n = bias_n
        self.momentum_day = momentum_day
        self.slope_n = slope_n
        self.hold_period = hold_period
        self.rebalance_threshold = rebalance_threshold

        self.days_counter = 0
        self.ohlc_history: dict = {}   # {symbol: [dict]}
        self.last_valid_ohlc: dict = {}
        self.trade_counter = 0

    def preload_history(self, history_map: dict):
        """预加载 OHLC 数据"""
        for symbol, df in history_map.items():
            if not df.empty:
                records = df[["open", "high", "low", "close"]].to_dict("records")
                self.ohlc_history[symbol] = records
                if records:
                    self.last_valid_ohlc[symbol] = records[-1]
                self.history[symbol] = df["close"].tolist()

    def on_bar(self, date, bars: dict):
        self._update_ohlc(bars)
        self.days_counter += 1
        if self.days_counter % self.hold_period != 0:
            return
        self._rebalance(date, bars)

    # ──────────────────── 数据维护 ────────────────────

    def _update_ohlc(self, bars: dict):
        """逐日更新 OHLC 历史, 停牌日用前值填充"""
        for symbol in self.symbols:
            if symbol in bars:
                bar = bars[symbol]
                record = {k: bar[k] for k in ("open", "high", "low", "close")}
                self.ohlc_history.setdefault(symbol, []).append(record)
                self.last_valid_ohlc[symbol] = record
                self.history.setdefault(symbol, []).append(bar["close"])
            elif symbol in self.last_valid_ohlc:
                fill = self.last_valid_ohlc[symbol]
                self.ohlc_history[symbol].append(fill)
                self.history[symbol].append(fill["close"])

    # ──────────────────── 三因子计算 ────────────────────

    def _bias_factor(self, closes: list) -> float | None:
        """乖离动量: BIAS 均线偏离 → 线性回归斜率"""
        required = self.bias_n + self.momentum_day
        if len(closes) < required:
            return None
        s = pd.Series(closes)
        ma = s.rolling(window=self.bias_n, min_periods=1).mean()
        bias = s / ma
        recent = bias.iloc[-self.momentum_day:]
        if recent.iloc[0] == 0:
            return 0.0
        y = (recent / recent.iloc[0]).values
        try:
            slope, _, _, _, _ = stats.linregress(np.arange(len(y)), y)
            return slope * 10000
        except Exception:
            return 0.0

    def _slope_factor(self, closes: list) -> float | None:
        """斜率动量: 标准化价格的回归斜率 × R²"""
        if len(closes) < self.slope_n:
            return None
        arr = np.array(closes[-self.slope_n:])
        if arr[0] == 0:
            return 0.0
        normalized = arr / arr[0]
        try:
            slope, _, r_value, _, _ = stats.linregress(np.arange(1, len(arr) + 1), normalized)
            return 10000 * slope * (r_value ** 2)
        except Exception:
            return 0.0

    def _efficiency_factor(self, df: pd.DataFrame) -> float | None:
        """效率动量: 净方向移动 / 总波动路径"""
        if len(df) < self.slope_n:
            return None
        sub = df.iloc[-self.slope_n:].copy()
        sub["pivot"] = (sub["open"] + sub["high"] + sub["low"] + sub["close"]) / 4.0
        if sub["pivot"].min() <= 0:
            return 0.0
        log_pivot = np.log(sub["pivot"])
        momentum = 100 * (log_pivot.iloc[-1] - log_pivot.iloc[0])
        direction = abs(log_pivot.iloc[-1] - log_pivot.iloc[0])
        volatility = log_pivot.diff().abs().sum()
        ratio = direction / volatility if volatility > 1e-6 else 0.0
        return momentum * ratio

    # ──────────────────── 调仓逻辑 ────────────────────

    def _rebalance(self, date, bars: dict):
        factor_rows = []
        min_len = max(self.bias_n + self.momentum_day, self.slope_n)

        for symbol in self.symbols:
            if symbol not in self.ohlc_history or len(self.ohlc_history[symbol]) < min_len:
                continue
            closes = [r["close"] for r in self.ohlc_history[symbol]]
            df_hist = pd.DataFrame(self.ohlc_history[symbol])
            b = self._bias_factor(closes)
            s = self._slope_factor(closes)
            e = self._efficiency_factor(df_hist)
            if b is not None and s is not None and e is not None:
                factor_rows.append({"symbol": symbol, "bias": b, "slope": s, "efficiency": e})

        if not factor_rows:
            return

        df = pd.DataFrame(factor_rows).set_index("symbol")
        stds = df.std().replace(0, 1)
        z = (df - df.mean()) / stds
        z["total"] = z.sum(axis=1)
        ranked = z.sort_values("total", ascending=False)

        logger.info(f"\n{date} 调仓 - Top 候选:\n{ranked[['total']].head(5)}")

        if ranked.empty:
            return

        best_symbol = ranked.index[0]
        best_score = ranked.iloc[0]["total"]
        current_positions = list(self.broker.positions.keys())
        held = current_positions[0] if current_positions else None
        target = best_symbol

        if held:
            if held not in ranked.index:
                target = best_symbol
            elif held == best_symbol:
                target = held
            else:
                current_score = ranked.loc[held, "total"]
                if best_score > current_score * self.rebalance_threshold:
                    logger.info(f"切换 {held}→{best_symbol}: "
                                f"{best_score:.2f} > {current_score:.2f}×{self.rebalance_threshold}")
                    target = best_symbol
                else:
                    logger.info(f"保持 {held} ({current_score:.2f})")
                    target = held

        # 执行交易
        for s in list(self.broker.positions.keys()):
            if s != target and s in bars:
                self.broker.sell(date, s, bars[s]["close"],
                                 self.broker.positions[s].quantity)

        if target not in self.broker.positions and target in bars:
            price = bars[target]["close"]
            if price > 0:
                cr = self.broker.commission_rate
                qty = int(self.broker.cash / (price * (1 + cr)) // self.lot_size) * self.lot_size
                if qty > 0:
                    self.broker.buy(date, target, price, qty)
