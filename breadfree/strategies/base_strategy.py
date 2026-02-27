"""策略基类 — 所有策略继承自 BreadFreeStrategy"""

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BreadFreeStrategy:
    """策略接口基类, 定义 set_symbols / preload_history / on_bar 三个生命周期方法."""

    def __init__(self, broker, lot_size: int = 100):
        self.broker = broker
        self.lot_size = lot_size
        self.symbols: list = []
        self.history: dict = {}  # {symbol: [close_price, ...]}

    def set_symbols(self, symbols: list):
        """设置标的池并初始化价格历史容器"""
        self.symbols = symbols
        for s in symbols:
            if s not in self.history:
                self.history[s] = []

    def preload_history(self, history_map: dict):
        """预加载历史数据 (warmup 阶段调用)

        Args:
            history_map: {symbol: DataFrame} — 必须包含 'close' 列
        """
        for symbol, df in history_map.items():
            if not df.empty and "close" in df.columns:
                self.history[symbol] = df["close"].tolist()
                logger.info(f"{self.__class__.__name__}: 预加载 {symbol} 共 "
                            f"{len(self.history[symbol])} 天")

    def on_bar(self, date, bars: dict):
        """每个交易日回调 — 子类必须实现

        Args:
            date: 当前日期
            bars: {symbol: bar_data} — bar_data 包含 open/high/low/close/volume
        """
        raise NotImplementedError
