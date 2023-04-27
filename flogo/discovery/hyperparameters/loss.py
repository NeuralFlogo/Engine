class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def compute(self, preds, labels):
        self.loss_function.compute(preds, labels)
