class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def compute(self, preds, labels):
        return self.loss_function.compute(preds, labels)
