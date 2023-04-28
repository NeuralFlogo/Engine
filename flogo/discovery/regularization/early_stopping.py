class EarlyStopping:
    def __init__(self, *monitors):
        self.monitors = monitors

    def check(self, accuracy, loss):
        return all([monitor.monitor(accuracy=accuracy, loss=loss) for monitor in self.monitors])
