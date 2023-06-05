from math import inf


class GrowthMonitor:
    def __init__(self, patience: int, improvement_threshold: float, metric="Accuracy"):
        self.improvement_threshold = improvement_threshold
        self.metric = metric
        self.history = self.__init_history(patience)

    def __init_history(self, patience):
        return [0] * (patience + 1) if self.metric == "Accuracy" else [inf] * (patience + 1)

    def supervise(self, value) -> bool:
        self.history.append(value)
        return abs(self.__find_edge_value() - self.history.pop(0)) < self.improvement_threshold

    def __find_edge_value(self):
        return max(self.history) if self.metric == "Accuracy" else min(self.history)

