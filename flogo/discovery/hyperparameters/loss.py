class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def compute(self, predictions, labels):
        return self.loss_function.measure(predictions, labels)
