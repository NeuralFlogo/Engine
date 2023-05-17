class TestTask:
    def __init__(self, dataset, measurer, tester):
        self.tester = self.__init_tester(tester, dataset, measurer)

    def __init_tester(self, tester, dataset, measurer):
        return tester(dataset, measurer)

    def execute(self, model):
        return self.tester.test(model)
