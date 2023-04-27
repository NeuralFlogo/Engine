class Optimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()
