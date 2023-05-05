class PrecisionMonitor:
    def __init__(self, threshold: int):
        self.threshold = threshold

    def monitor(self, loss, accuracy) -> bool:
        return accuracy < self.threshold
