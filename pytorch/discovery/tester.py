import torch


class PytorchTester:

    def __init__(self, dataset, measurer):
        self.dataset = dataset
        self.measurer = measurer

    def test(self, model):
        measure = 0
        with torch.no_grad():
            for entry in self.dataset:
                inputs, expected = entry.get_input(), entry.get_output()
                measure += self.measurer.measure(self.__evaluate(model, inputs), expected)
        return self.__average_quality(measure)

    def __average_quality(self, measure):
        return measure / len(self.dataset)

    def __evaluate(self, model, inputs):
        return model(inputs)
