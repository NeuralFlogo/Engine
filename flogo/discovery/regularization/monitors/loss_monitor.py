from math import inf


class LossMonitor:
    def __init__(self, patience: int, improvement_threshold: float):
        self.improvement_threshold = improvement_threshold
        self.history = [inf] * (patience + 1)

    def monitor(self, accuracy, loss) -> bool:
        self.history.append(loss)
        return self.history.pop(0) - min(self.history) >= self.improvement_threshold
