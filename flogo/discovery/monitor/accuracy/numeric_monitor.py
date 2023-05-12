CEILING = 100


class NumericMonitor:
    def compute(self, predictions, outputs):
        return self.__compute_error(predictions, outputs) * CEILING / outputs

    def __compute_error(self, predictions, outputs):
        return outputs - predictions
