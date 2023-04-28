class AccuracyMonitor:
    def __init__(self, patience: int, improvement_threshold: float):
        self.improvement_threshold = improvement_threshold
        self.history = [0] * (patience + 1)

    def monitor(self, accuracy, loss) -> bool:
        self.history.append(accuracy)
        return max(self.history) - self.history.pop(0) >= self.improvement_threshold
