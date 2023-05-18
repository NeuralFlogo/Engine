class PrecisionMonitor:
    def __init__(self, threshold: int):
        self.threshold = threshold

    def supervise(self, value) -> bool:
        return value == self.threshold
