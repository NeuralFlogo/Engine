class TestTask:
    def __init__(self, tester):
        self.tester = tester

    def execute(self, model):
        return self.tester.test(model)
