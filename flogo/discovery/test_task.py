from pytorch.discovery.test_task import PytorchTestTask


class TestTask:
    def __init__(self, dataset, task: PytorchTestTask.__class__):
        self.dataset = dataset
        self.task = task

    def test(self, model):
        return self.task(self.dataset).execute(model)
