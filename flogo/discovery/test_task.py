from pytorch.discovery.test_task import PytorchTestTask


class TestTask:
    def __init__(self, model, dataset, task: PytorchTestTask.__class__):
        self.model = model
        self.dataset = dataset
        self.task = task

    def test(self):
        self.task(self.model, self.dataset).execute()
