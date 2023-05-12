from pytorch.discovery.tester import PytorchTester


class TestTask:
    def __init__(self, dataset, measurer, tester: PytorchTester.__class__):
        self.tester = self.__init_tester(tester, dataset, measurer)

    def __init_tester(self, tester, dataset, measurer):
        return tester(dataset, measurer)

    def execute(self, model):
        return self.tester.test(model)
