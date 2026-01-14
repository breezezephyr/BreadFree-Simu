from .base_strategy import BreadFreeStrategy
import pandas as pd
import numpy as np
from scipy import stats
from ..utils.logger import get_logger
from collections import defaultdict

logger = get_logger(__name__)

class RotationStrategy(BreadFreeStrategy):
    def __init__(self, broker, lookback_period=20, hold_period=20, top_n=2,
                 use_efficiency=True, lot_size=100, min_data_ratio=1.5,
                 enable_dynamic_weight=False, min_momentum=0.0):
        """
        增强版ETF效率轮动策略
        :param broker: Broker实例
        :param lookback_period: 回顾周期（天）
        :param hold_period: 持仓周期（天）
        :param top_n: 持有前N只ETF
        :param use_efficiency: True → 效率轮动；False → 纯动量轮动
        :param lot_size: 最小交易单位
        :param min_data_ratio: 最小数据比率（相对lookback_period）
        :param enable_dynamic_weight: 启用动态权重分配
        :param min_momentum: 动量筛选阈值（0表示不筛选）
        """
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.top_n = top_n
        self.use_efficiency = use_efficiency
        self.min_data_ratio = min_data_ratio
        self.min_data_length = max(int(lookback_period * min_data_ratio), 30)
        self.enable_dynamic_weight = enable_dynamic_weight
        self.min_momentum = min_momentum
        
        # 增强型数据结构
        self.days_counter = 0
        self.suspension_flags = defaultdict(bool)
        self.last_valid_prices = {}
        
        # 性能监控
        self.trade_counter = 0
        self.rebalance_dates = []

    def stable_linear_regression(self, prices):
        """稳健的线性回归计算（带异常值处理）"""
        n = len(prices)
        if n < 2:
            return 0.0, 0.0, 0.0  # slope, intercept, r2
        
        try:
            x = np.arange(n)
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
            r2 = r_value ** 2
            return slope, intercept, r2
        except:
            # 回归失败时使用简单方法
            price_diff = prices[-1] - prices[0]
            return price_diff / max(n-1, 1), prices[0], 0.0

    def calculate_efficiency_score(self, symbol, window):
        """
        计算效率得分：(动量 / 波动率) * R²（趋势稳定性）
        使用向量化计算提高性能
        """
        history = self.history[symbol]
        
        # 数据长度检查
        if len(history) < self.min_data_length:
            logger.debug(f"数据不足: {symbol} ({len(history)} < {self.min_data_length})")
            return -np.inf
            
        # 获取有效窗口数据
        start_idx = max(0, len(history) - window)
        prices = np.array(history[start_idx:])
        
        # 1. 动量 (ROC)
        momentum = (prices[-1] / prices[0] - 1) if prices[0] > 0 else 0
        
        # 纯动量模式直接返回
        if not self.use_efficiency:
            return momentum
            
        try:
            # 2. 波动率 (Std of returns)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # 3. 趋势稳定性 (R^2)
            _, _, r2 = self.stable_linear_regression(prices)
            
            # 4. 效率得分计算
            epsilon = 1e-6
            efficiency = (momentum / (volatility + epsilon)) * max(r2, 0)
            return efficiency
        except Exception as e:
            logger.error(f"计算指标错误 {symbol}: {e}")
            return -np.inf

    def update_suspension_status(self, date, bars):
        """更新停牌状态并填充历史价格"""
        for symbol in self.symbols:
            # 检测停牌：当日无数据但前日有数据
            if symbol not in bars and symbol in self.last_valid_prices:
                self.suspension_flags[symbol] = True
                last_price = self.last_valid_prices[symbol]
                self.history[symbol].append(last_price)
                logger.debug(f"{date} {symbol} 停牌，使用前价: {last_price}")
            elif symbol in bars:
                price = bars[symbol]['close']
                self.history[symbol].append(price)
                self.last_valid_prices[symbol] = price
                self.suspension_flags[symbol] = False
            else:
                # 新上市或长期停牌
                if symbol not in self.history or not self.history[symbol]:
                    self.history[symbol] = []

    def filter_valid_symbols(self):
        """筛选有效标的"""
        valid_symbols = []
        for symbol in self.symbols:
            # 排除停牌且数据不足的
            if self.suspension_flags.get(symbol, False):
                continue
                
            # 检查数据长度
            if len(self.history.get(symbol, [])) < self.min_data_length:
                continue
                
            valid_symbols.append(symbol)
            
        return valid_symbols

    def calculate_weights(self, scores, symbols):
        """计算动态权重分配"""
        if not self.enable_dynamic_weight or len(symbols) == 0:
            # 等权重分配
            return {s: 1.0/len(symbols) for s in symbols}
            
        # 基于分数的比例分配（softmax）
        score_values = np.array([scores[s] for s in symbols])
        exp_scores = np.exp(score_values - np.max(score_values))
        weights = exp_scores / exp_scores.sum()
        return {s: w for s, w in zip(symbols, weights)}

    def execute_rebalancing(self, date, bars, target_symbols, scores):
        """执行再平衡交易"""
        # 1. 计算当前总权益
        current_prices = {s: bars[s]['close'] for s in bars if s in self.history}
        total_equity = self.broker.get_total_equity(current_prices)
        
        # 2. 计算目标权重
        target_weights = self.calculate_weights(scores, target_symbols)
        logger.info(f"目标权重: {target_weights}")
        
        # 3. 确定目标持仓价值
        target_values = {s: total_equity * w for s, w in target_weights.items()}
        
        # 4. 卖出非目标持仓
        current_positions = list(self.broker.positions.keys())
        for symbol in current_positions:
            if symbol not in target_symbols and not self.suspension_flags.get(symbol, False):
                if symbol in bars:
                    pos = self.broker.positions[symbol]
                    if pos.quantity > 0:  # 只在有持仓时才卖出和记录日志
                        self.broker.sell(date, symbol, bars[symbol]['close'], pos.quantity)
                        logger.info(f"卖出 {symbol}: {pos.quantity}股")
                        self.trade_counter += 1
                else:
                    logger.warning(f"无法卖出 {symbol} (停牌?)")

        # 5. 买入/调整目标持仓
        for symbol in target_symbols:
            if symbol not in bars:
                logger.warning(f"跳过买入 {symbol} (停牌?)")
                continue
                
            price = bars[symbol]['close']
            target_value = target_values[symbol]
            
            # 当前持仓价值
            current_value = 0
            if symbol in self.broker.positions:
                pos = self.broker.positions[symbol]
                current_value = pos.quantity * price
                
            # 需要调整的价值差
            value_diff = target_value - current_value
            
            # 买入逻辑
            if value_diff > 0:
                # 计算考虑手续费的可用资金
                available_cash = self.broker.cash
                max_buy_value = available_cash / (1 + self.broker.commission_rate)
                buy_value = min(value_diff, max_buy_value)
                
                # 计算交易数量（考虑最小交易单位）
                buy_qty = int(buy_value // price)
                buy_qty = (buy_qty // self.lot_size) * self.lot_size
                
                if buy_qty > 0:
                    self.broker.buy(date, symbol, price, buy_qty)
                    logger.info(f"买入 {symbol}: {buy_qty}股 @ {price} (目标权重: {target_weights[symbol]:.2%})")
                    self.trade_counter += 1
            
            # 卖出逻辑（持仓超过目标）
            elif value_diff < 0:
                sell_value = -value_diff
                sell_qty = int(sell_value // price)
                # 不能卖出超过当前持仓
                if symbol in self.broker.positions:
                    current_qty = self.broker.positions[symbol].quantity
                    sell_qty = min(sell_qty, current_qty)
                    sell_qty = (sell_qty // self.lot_size) * self.lot_size
                    
                    if sell_qty > 0:
                        self.broker.sell(date, symbol, price, sell_qty)
                        logger.info(f"减持 {symbol}: {sell_qty}股 (目标权重: {target_weights[symbol]:.2%})")
                        self.trade_counter += 1

    def on_bar(self, date, bars):
        # 1. 更新数据并处理停牌
        self.update_suspension_status(date, bars)
        self.days_counter += 1
        
        # 2. 检查调仓日
        if self.days_counter % self.hold_period != 0:
            return
            
        self.rebalance_dates.append(date)
        logger.info(f"\n{'='*50}\n{date} 调仓日 ({'效率轮动' if self.use_efficiency else '动量轮动'})")
        logger.info(f"当前持仓: {list(self.broker.positions.keys())}")
        
        # 3. 筛选有效标的
        valid_symbols = self.filter_valid_symbols()
        logger.info(f"有效标的: {len(valid_symbols)}/{len(self.symbols)}")
        
        # 4. 计算得分
        scores = {}
        for symbol in valid_symbols:
            scores[symbol] = self.calculate_efficiency_score(symbol, self.lookback_period)
        
        # 5. 筛选目标标的
        # 移除无效分数
        valid_scores = {k: v for k, v in scores.items() if v > -np.inf}
        
        # 按分数排序
        sorted_symbols = sorted(valid_scores.keys(), key=lambda x: valid_scores[x], reverse=True)
        
        # 动量阈值筛选
        if self.min_momentum > 0:
            target_symbols = [s for s in sorted_symbols[:self.top_n] 
                             if valid_scores[s] >= self.min_momentum]
            if len(target_symbols) < self.top_n:
                logger.info(f"仅有{len(target_symbols)}只标的满足动量阈值")
        else:
            target_symbols = sorted_symbols[:self.top_n]
        
        logger.info(f"目标持仓: {target_symbols}")
        
        # 6. 执行再平衡
        if target_symbols:
            self.execute_rebalancing(date, bars, target_symbols, scores)
        else:
            logger.info("无符合条件标的，保持现金")
        # 记录持仓状态
        current_prices = {s: bars[s]['close'] for s in bars if s in self.history}
        logger.info(f"调仓后资产: {self.broker.get_total_equity(current_prices):.2f}")
        
    def get_performance_metrics(self):
        """获取策略性能指标"""
        return {
            'total_trades': self.trade_counter,
            'rebalance_count': len(self.rebalance_dates),
            'last_rebalance': self.rebalance_dates[-1] if self.rebalance_dates else None
        }
