class EarlyStopping:
    def __init__(self, monitor):
        self.monitor = monitor

    def check(self, measure):
        return self.monitor.supervise(measure)
