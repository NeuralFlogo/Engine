class PrecisionMonitor:
    def __init__(self, threshold: int):
        self.threshold = threshold

    def monitor(self, loss, accuracy):
        return accuracy < self.threshold
