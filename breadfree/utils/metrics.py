import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import stats

def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """
    计算最大回撤
    :param equity_series: 净值序列
    :return: 最大回撤 (负数)
    """
    if equity_series.empty:
        return 0.0
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return float(drawdown.min())

def calculate_sharpe_ratio(equity_series: pd.Series, risk_free_rate: float = 0.02, annual_days: int = 242) -> float:
    """
    计算夏普比率 (基于日收益率)
    :param equity_series: 净值序列
    :param risk_free_rate: 无风险收益率 (年化)，默认 2%
    :param annual_days: 交易日个数，A股通常为 242 左右
    :return: 夏普比率
    """
    if len(equity_series) < 2:
        return 0.0
    
    # 计算日收益率
    daily_returns = equity_series.pct_change().dropna()
    
    if daily_returns.std() == 0:
        return 0.0
        
    # 日均超额收益
    daily_rf = risk_free_rate / annual_days
    excess_returns = daily_returns - daily_rf
    
    # 年化收益 / 年化波动
    sharpe = (excess_returns.mean() / daily_returns.std()) * (annual_days ** 0.5)
    return float(sharpe)

def calculate_total_return(equity_series: pd.Series, initial_capital: float = None) -> float:
    """
    计算总收益率
    :param equity_series: 净值序列
    :param initial_capital: 初始资金 (可选，如果提供则基于初始资金计算，否则基于序列首项)
    """
    if equity_series.empty:
        return 0.0
    
    final_value = equity_series.iloc[-1]
    start_value = initial_capital if initial_capital is not None else equity_series.iloc[0]
    
    if start_value == 0:
        return 0.0
        
    return (final_value - start_value) / start_value

def calculate_annualized_return(equity_series: pd.Series, annual_days: int = 242) -> float:
    """
    计算年化收益率
    """
    if len(equity_series) < 2:
        return 0.0
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    num_days = len(equity_series)
    if num_days == 0:
        return 0.0
        
    annualized_return = (1 + total_return) ** (annual_days / num_days) - 1
    return float(annualized_return)

def calculate_calmar_ratio(annual_return: float, max_drawdown: float, risk_free_rate: float = 0.02) -> float:
    """
    计算卡玛比率 (年化收益 / 最大回撤)
    :param annual_return: 年化收益率
    :param max_drawdown: 最大回撤（绝对值，正数）
    :param risk_free_rate: 无风险收益率
    :return: 卡玛比率
    """
    if max_drawdown == 0:
        return 0.0
    return float((annual_return - risk_free_rate) / abs(max_drawdown))

def calculate_sortino_ratio(equity_series: pd.Series, risk_free_rate: float = 0.02,
                           annual_days: int = 242, target_return: float = 0.0) -> float:
    """
    计算索提诺比率 (考虑下行风险)
    :param equity_series: 净值序列
    :param risk_free_rate: 无风险收益率
    :param annual_days: 年交易日数
    :param target_return: 目标收益率 (默认0)
    :return: 索提诺比率
    """
    if len(equity_series) < 2:
        return 0.0
    
    daily_returns = equity_series.pct_change().dropna()
    daily_target = target_return / annual_days
    daily_rf = risk_free_rate / annual_days
    
    # 计算下行偏差 (只考虑低于目标收益的部分)
    downside_returns = daily_returns[daily_returns < daily_target] - daily_target
    if len(downside_returns) == 0:
        downside_std = 0
    else:
        downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0.0
    
    annualized_excess_return = (daily_returns.mean() - daily_rf) * annual_days
    sortino_ratio = annualized_excess_return / (downside_std * np.sqrt(annual_days))
    
    return float(sortino_ratio)

def calculate_information_ratio(portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series,
                               annual_days: int = 242) -> float:
    """
    计算信息比率 (超额收益的稳定性)
    :param portfolio_returns: 投资组合收益率序列 (日频)
    :param benchmark_returns: 基准收益率序列 (日频)
    :param annual_days: 年交易日数
    :return: 信息比率
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0
    
    excess_returns = portfolio_returns - benchmark_returns
    if excess_returns.std() == 0:
        return 0.0
    
    info_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annual_days)
    return float(info_ratio)

def calculate_alpha_beta(portfolio_returns: pd.Series,
                        benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.02,
                        annual_days: int = 242) -> Tuple[float, float]:
    """
    计算Alpha和Beta系数
    :return: (alpha年化, beta)
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0, 1.0
    
    daily_rf = risk_free_rate / annual_days
    excess_portfolio = portfolio_returns - daily_rf
    excess_benchmark = benchmark_returns - daily_rf
    
    # 计算Beta (协方差 / 方差)
    covariance = np.cov(excess_portfolio, excess_benchmark)[0, 1]
    benchmark_var = np.var(excess_benchmark)
    
    if benchmark_var == 0:
        beta = 1.0
    else:
        beta = covariance / benchmark_var
    
    # 计算Alpha
    alpha_daily = excess_portfolio.mean() - beta * excess_benchmark.mean()
    alpha_annual = alpha_daily * annual_days
    
    return float(alpha_annual), float(beta)

def calculate_tracking_error(portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series,
                           annual_days: int = 242) -> float:
    """
    计算跟踪误差 (年化)
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(annual_days)
    
    return float(tracking_error)

def calculate_win_rate(trade_returns: List[float]) -> Tuple[float, int, int]:
    """
    计算胜率、胜场数、总交易数
    :param trade_returns: 每笔交易的收益率列表
    :return: (胜率, 胜场数, 总交易数)
    """
    if not trade_returns:
        return 0.0, 0, 0
    
    wins = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = wins / total if total > 0 else 0.0
    
    return float(win_rate), wins, total

def calculate_profit_factor(gross_profits: float, gross_losses: float) -> float:
    """
    计算盈利因子 (总盈利 / 总亏损)
    """
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    return float(gross_profits / abs(gross_losses))

def calculate_average_win_loss(trade_returns: List[float]) -> Tuple[float, float]:
    """
    计算平均盈利和平均亏损
    """
    if not trade_returns:
        return 0.0, 0.0
    
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    return float(avg_win), float(avg_loss)

def calculate_max_consecutive_wins_losses(trade_returns: List[float]) -> Tuple[int, int]:
    """
    计算最大连续盈利和最大连续亏损次数
    """
    if not trade_returns:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for ret in trade_returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif ret < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    return max_wins, max_losses

def calculate_r_squared(portfolio_returns: pd.Series, 
                       benchmark_returns: pd.Series) -> float:
    """
    计算R平方 (拟合优度)
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 0.0
    
    # 使用线性回归计算R平方
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        benchmark_returns, portfolio_returns
    )
    
    return float(r_value ** 2)

def calculate_volatility(returns_series: pd.Series, annual_days: int = 242) -> float:
    """
    计算年化波动率
    """
    if len(returns_series) < 2:
        return 0.0
    
    return float(returns_series.std() * np.sqrt(annual_days))

def calculate_value_at_risk(returns_series: pd.Series, 
                           confidence_level: float = 0.95,
                           method: str = 'historical') -> float:
    """
    计算在险价值 (VaR)
    :param confidence_level: 置信水平
    :param method: 计算方法 - 'historical'历史法, 'parametric'参数法
    :return: VaR值 (负数表示损失)
    """
    if len(returns_series) < 2:
        return 0.0
    
    if method == 'historical':
        # 历史模拟法
        var = np.percentile(returns_series, (1 - confidence_level) * 100)
    else:
        # 参数法 (正态分布假设)
        mean = returns_series.mean()
        std = returns_series.std()
        from scipy.stats import norm
        var = norm.ppf(1 - confidence_level, mean, std)
    
    return float(var)

def calculate_conditional_var(returns_series: pd.Series,
                            confidence_level: float = 0.95) -> float:
    """
    计算条件在险价值 (CVaR/Expected Shortfall)
    即超过VaR部分的平均损失
    """
    if len(returns_series) < 2:
        return 0.0
    
    var = calculate_value_at_risk(returns_series, confidence_level, 'historical')
    cvar = returns_series[returns_series <= var].mean()
    
    return float(cvar)

def calculate_skewness_kurtosis(returns_series: pd.Series) -> Tuple[float, float]:
    """
    计算收益率序列的偏度和峰度
    """
    if len(returns_series) < 2:
        return 0.0, 0.0
    
    skewness = float(returns_series.skew())
    kurtosis = float(returns_series.kurtosis())  # 超额峰度
    
    return skewness, kurtosis

def calculate_jensen_alpha(portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          risk_free_rate: float = 0.02,
                          annual_days: int = 242) -> float:
    """
    计算詹森阿尔法 (基于CAPM模型)
    """
    alpha, beta = calculate_alpha_beta(
        portfolio_returns, benchmark_returns, risk_free_rate, annual_days
    )
    return alpha

def calculate_ulcer_index(equity_series: pd.Series) -> float:
    """
    计算溃疡指数 (衡量下跌波动性)
    """
    if len(equity_series) < 2:
        return 0.0
    
    running_max = equity_series.cummax()
    drawdown = ((equity_series - running_max) / running_max) * 100
    squared_dd = drawdown ** 2
    
    ulcer_index = np.sqrt(squared_dd.mean())
    return float(ulcer_index)

def calculate_omega_ratio(returns_series: pd.Series,
                         threshold: float = 0.0) -> float:
    """
    计算Omega比率 (收益高于阈值的概率加权)
    """
    if len(returns_series) < 2:
        return 0.0
    
    gains = returns_series[returns_series > threshold] - threshold
    losses = threshold - returns_series[returns_series < threshold]
    
    if len(losses) == 0 or losses.sum() == 0:
        return float('inf')
    
    omega = gains.sum() / losses.sum()
    return float(omega)

def calculate_all_metrics(equity_series: pd.Series,
                         trade_returns: List[float] = None,
                         benchmark_returns: pd.Series = None,
                         risk_free_rate: float = 0.02,
                         annual_days: int = 242) -> Dict:
    """
    计算所有指标的综合函数
    """
    metrics = {}
    
    # 基本收益指标
    metrics['total_return'] = calculate_total_return(equity_series)
    metrics['annualized_return'] = calculate_annualized_return(equity_series, annual_days)
    metrics['max_drawdown'] = calculate_max_drawdown(equity_series)
    
    # 风险调整收益指标
    daily_returns = equity_series.pct_change().dropna() if len(equity_series) > 1 else pd.Series()
    if not daily_returns.empty:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(equity_series, risk_free_rate, annual_days)
        metrics['sortino_ratio'] = calculate_sortino_ratio(equity_series, risk_free_rate, annual_days)
        metrics['calmar_ratio'] = calculate_calmar_ratio(equity_series, risk_free_rate, annual_days)
        metrics['volatility'] = calculate_volatility(daily_returns, annual_days)
        
        # 分布特征
        metrics['skewness'], metrics['kurtosis'] = calculate_skewness_kurtosis(daily_returns)
        
        # 风险指标
        metrics['var_95'] = calculate_value_at_risk(daily_returns, 0.95)
        metrics['cvar_95'] = calculate_conditional_var(daily_returns, 0.95)
        metrics['ulcer_index'] = calculate_ulcer_index(equity_series)
        metrics['omega_ratio'] = calculate_omega_ratio(daily_returns)
    
    # 交易统计
    if trade_returns:
        win_rate, wins, total = calculate_win_rate(trade_returns)
        metrics['win_rate'] = win_rate
        metrics['total_trades'] = total
        metrics['profit_trades'] = wins
        metrics['loss_trades'] = total - wins
        
        avg_win, avg_loss = calculate_average_win_loss(trade_returns)
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        
        if avg_loss != 0:
            metrics['win_loss_ratio'] = abs(avg_win / avg_loss)
        
        # 计算总盈利和总亏损
        gross_profits = sum(r for r in trade_returns if r > 0)
        gross_losses = sum(r for r in trade_returns if r < 0)
        metrics['gross_profit'] = gross_profits
        metrics['gross_loss'] = gross_losses
        metrics['profit_factor'] = calculate_profit_factor(gross_profits, gross_losses)
        
        # 连续统计
        max_wins, max_losses = calculate_max_consecutive_wins_losses(trade_returns)
        metrics['max_consecutive_wins'] = max_wins
        metrics['max_consecutive_losses'] = max_losses
    
    # 相对基准指标
    if benchmark_returns is not None and not daily_returns.empty:
        metrics['information_ratio'] = calculate_information_ratio(
            daily_returns, benchmark_returns, annual_days
        )
        metrics['alpha'], metrics['beta'] = calculate_alpha_beta(
            daily_returns, benchmark_returns, risk_free_rate, annual_days
        )
        metrics['tracking_error'] = calculate_tracking_error(
            daily_returns, benchmark_returns, annual_days
        )
        metrics['r_squared'] = calculate_r_squared(daily_returns, benchmark_returns)
        metrics['jensen_alpha'] = calculate_jensen_alpha(
            daily_returns, benchmark_returns, risk_free_rate, annual_days
        )
    
    return metrics

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 模拟净值曲线
    returns = np.random.normal(0.0005, 0.02, 100)
    equity = 1000000 * (1 + returns).cumprod()
    equity_series = pd.Series(equity, index=dates)
    
    # 模拟交易记录
    trade_returns = np.random.normal(0.001, 0.05, 50).tolist()
    
    # 模拟基准
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, 100), index=dates)
    
    # 计算所有指标
    metrics = calculate_all_metrics(
        equity_series=equity_series,
        trade_returns=trade_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02,
        annual_days=242
    )
    
    # 打印结果
    print("=== 绩效指标汇总 ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")