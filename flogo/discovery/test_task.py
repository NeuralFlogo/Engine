from pytorch.discovery.tester import PytorchTester


class TestTask:
    def __init__(self, dataset, task: PytorchTester.__class__):
        self.dataset = dataset
        self.task = task

    def test(self, model):
        return self.task(self.dataset).execute(model)
