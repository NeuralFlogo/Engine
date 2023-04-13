class FlogoOptimizer:
    def __init__(self, name: str, model_params: iter, lr: float):
        self.name = name
        self.model_params = model_params
        self.lr = lr
