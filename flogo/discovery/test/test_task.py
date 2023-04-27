class TestTask:
    def __init__(self, test_strategy):
        self.test_strategy = test_strategy

    def test(self):
        self.test_strategy.apply()
