from pytorch.discovery.tester import PytorchTester


class TestTask:
    def __init__(self, dataset, measurer, task: PytorchTester.__class__):
        self.task = self.__init_task(task, dataset, measurer)

    def __init_task(self, task, dataset, measurer):
        return task(dataset, measurer)

    def test(self, model):
        return self.task.execute(model)
