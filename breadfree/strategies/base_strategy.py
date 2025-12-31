class Strategy:
    def __init__(self, broker, lot_size=100):
        self.broker = broker
        self.lot_size = lot_size

    def on_bar(self, date, bar_data):
        """
        Called on every bar (day)
        :param date: current date
        :param bar_data: dict or series containing Open, High, Low, Close, Volume
        """
        raise NotImplementedError
