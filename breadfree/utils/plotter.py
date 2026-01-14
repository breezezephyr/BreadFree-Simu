import pandas as pd
from pyecharts.charts import Line, Grid
from pyecharts import options as opts
from .metrics import calculate_max_drawdown, calculate_sharpe_ratio

def plot_backtest_results(equity_curve, transaction_history, benchmark_series=None, initial_cash=100000.0, title="Backtest Result", filename="backtest_result.html"):
    """
    绘制回测结果图表 (支持多股/资产组合)
    
    Args:
        equity_curve: list of dict, e.g. [{'date': datetime, 'equity': float}, ...]
        transaction_history: list of dict, e.g. [{'date': datetime, 'action': 'BUY', ...}, ...]
        benchmark_series: pd.Series, index为日期, values为基准价格/净值. 可选.
        initial_cash: float, 初始资金
        title: str, 图表标题
        filename: str, 保存文件名
    """
    if not equity_curve:
        print("No results to plot.")
        return

    # 1. 准备策略净值数据
    equity_values = [d['equity'] for d in equity_curve]
    # 确保日期格式化为字符串
    dates = [d['date'].strftime('%Y-%m-%d') if hasattr(d['date'], 'strftime') else str(d['date']) for d in equity_curve]
    
    equity_series = pd.Series(equity_values, index=pd.to_datetime(dates))

    # 2. 准备基准数据 (如果有)
    benchmark_equity = []
    if benchmark_series is not None and not benchmark_series.empty:
        # 对齐日期
        # benchmark_series 的索引应该是 datetime 类型
        if not isinstance(benchmark_series.index, pd.DatetimeIndex):
            benchmark_series.index = pd.to_datetime(benchmark_series.index)
            
        common_dates = benchmark_series.index.intersection(equity_series.index)
        
        if not common_dates.empty:
            # 截取对应时间段并归一化到初始资金
            aligned_benchmark = benchmark_series.loc[common_dates]
            # 按回测日期的顺序重新索引 (处理停牌或日期缺失情况，这里简单做交集处理)
            # 为了画图方便，我们在 dates 列表里找对应的基准值
            # 如果 dates 里的日期在 benchmark 里没有，就填充前值或者 None
            
            # 更简单的方法:
            # 取 benchmark 在 dates 范围内的切片，然后归一化
            start_price = 0
            # 找到第一个有效价格用来归一化
            for d in dates:
                dt = pd.to_datetime(d)
                if dt in benchmark_series.index:
                    start_price = benchmark_series.loc[dt]
                    break
            
            if start_price > 0:
                # 生成与 dates 对齐的 benchmark 列表
                for d in dates:
                    dt = pd.to_datetime(d)
                    if dt in benchmark_series.index:
                        val = benchmark_series.loc[dt]
                        benchmark_equity.append(val / start_price * initial_cash)
                    else:
                        # 如果缺失，沿用上一个值，或者 None
                        benchmark_equity.append(benchmark_equity[-1] if benchmark_equity else initial_cash)
            else:
                 benchmark_equity = [initial_cash] * len(dates)
        else:
            benchmark_equity = [initial_cash] * len(dates)
    else:
        # 无基准，填充 None 或不画
        pass

    # 3. 计算指标
    max_drawdown = calculate_max_drawdown(equity_series)
    sharpe_ratio = calculate_sharpe_ratio(equity_series)
    final_return = (equity_values[-1] - initial_cash) / initial_cash
    
    # 回撤曲线
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    
    subtitle = f"Total Return: {final_return:.2%} | Sharpe Ratio: {sharpe_ratio:.2f} | Max Drawdown: {max_drawdown:.2%}"

    # 4. 交易标记
    markers = []
    # 创建日期到净值的映射，方便查找坐标
    date_to_equity = dict(zip(dates, equity_values))
    
    for tx in transaction_history:
        tx_date = tx['date'].strftime('%Y-%m-%d') if hasattr(tx['date'], 'strftime') else str(tx['date'])
        
        # 只有当交易日期在图表的日期范围内时才标记
        if tx_date in date_to_equity:
            val = date_to_equity[tx_date]
            action = tx['action']
            
            # 多股情况下，可以在 tooltip 或 value 中显示 symbol
            symbol_info = tx.get('symbol', '')
            
            if action == 'BUY':
                markers.append(opts.MarkPointItem(
                    coord=[tx_date, val], 
                    value=f"{symbol_info}", 
                    symbol="arrow", 
                    symbol_size=15, 
                    itemstyle_opts=opts.ItemStyleOpts(color="#d14a61"),
                    # tooltip_opts=opts.TooltipOpts(formatter=f"Buy {symbol_info}<br/>Price: {tx['price']}")
                ))
            elif action == 'SELL':
                markers.append(opts.MarkPointItem(
                    coord=[tx_date, val], 
                    value=f"{symbol_info}", 
                    symbol="arrow", 
                    symbol_size=15, 
                    symbol_rotate=180, 
                    itemstyle_opts=opts.ItemStyleOpts(color="#5793f3"),
                    # tooltip_opts=opts.TooltipOpts(formatter=f"Sell {symbol_info}<br/>Price: {tx['price']}")
                ))

    # 5. 构建图表
    # 主图: 净值曲线 + 基准
    line_main = (
        Line()
        .add_xaxis(dates)
        .add_yaxis(
            "Strategy Equity", 
            equity_values, 
            is_smooth=True, 
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#c23531"),
            markpoint_opts=opts.MarkPointOpts(
                data=markers,
                label_opts=opts.LabelOpts(is_show=True)
            )
        )
    )
    
    if benchmark_equity and len(benchmark_equity) == len(dates):
        line_main.add_yaxis(
            "Benchmark", 
            benchmark_equity, 
            is_smooth=True, 
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#2f4554", type_="dashed")
        )

    line_main.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title, 
            subtitle=subtitle
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(
            type_="category", 
            boundary_gap=False, 
            axislabel_opts=opts.LabelOpts(is_show=False)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value", 
            is_scale=True, 
            axislabel_opts=opts.LabelOpts(formatter="{value}"),  # 这里改为 {value}
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        datazoom_opts=[opts.DataZoomOpts(xaxis_index=[0, 1], is_show=True)],
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )

    # 副图: 回撤
    line_drawdown = (
        Line()
        .add_xaxis(dates)
        .add_yaxis(
            "Drawdown", 
            drawdown.tolist(), 
            is_smooth=True, 
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#00da3c"),
            linestyle_opts=opts.LineStyleOpts(width=1, color="#00da3c"),
            yaxis_index=1,  # 新增，指定副图y轴
            xaxis_index=1   # 新增，指定副图x轴
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, grid_index=1),  # 新增 grid_index=1
            yaxis_opts=opts.AxisOpts(
                type_="value", 
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                grid_index=1  # 新增 grid_index=1
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    # 组合布局
    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="800px"))
        .add(line_main, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", height="60%"))
        .add(line_drawdown, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="75%", height="20%"))
    )

    grid.render(filename)
    return filename
