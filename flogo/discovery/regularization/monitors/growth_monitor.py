from math import inf


class GrowthMonitor:
    def __init__(self, patience: int, improvement_threshold: float, metric="Accuracy"):
        self.improvement_threshold = improvement_threshold
        self.history = self.__init_history(patience, metric)

    def __init_history(self, patience, metric):
        return [0] * (patience + 1) if metric == "Accuracy" else [inf] * (patience + 1)

    def supervise(self, value) -> bool:
        self.history.append(value)
        return max(self.history) - self.history.pop(0) < self.improvement_threshold

