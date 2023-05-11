CEILING = 100


class NumericMonitor:
    def compute(self, predictions, outputs):
        print("predictions: ", predictions.item())
        print("output: ", outputs.item())
        return self.__compute_error(predictions, outputs) * CEILING / outputs

    def __compute_error(self, predictions, outputs):
        return outputs - predictions
