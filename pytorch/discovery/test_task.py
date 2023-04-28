import torch


class PytorchTestTask:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def execute(self):
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.dataset):
                inputs, labels = data
                correct += self.__count_number_of_correct_predictions(inputs, labels)
        print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, len(self.dataset), self.__to_percentage(correct)))

    def __count_number_of_correct_predictions(self, inputs, labels):
        return torch.sum(torch.eq(self.find_index_with_greatest_value(self.__evaluate(inputs)),
                                  self.find_index_with_greatest_value(labels)))

    def __evaluate(self, inputs):
        return self.model(inputs)

    def find_index_with_greatest_value(self, values):
        return torch.argmax(values, dim=1)

    def __to_percentage(self, correct):
        return 100. * correct / len(self.dataset)

