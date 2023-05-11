import torch


class PytorchTestTask:

    def __init__(self, dataset):
        self.dataset = dataset

    def execute(self, model):
        correct = 0
        with torch.no_grad():
            for i, entry in enumerate(self.dataset):
                inputs, labels = entry.get_input(), entry.get_output()
                correct += self.__count_number_of_correct_predictions(self.__evaluate(model, inputs), labels)
        print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, len(self.dataset), self.__to_percentage(correct)))
        return self.__to_percentage(correct)

    def __count_number_of_correct_predictions(self, predictions, labels):
        return torch.sum(torch.eq(self.find_index_with_greatest_value(predictions),
                                  self.find_index_with_greatest_value(labels))).item()

    def __evaluate(self, model, inputs):
        return model(inputs)

    def find_index_with_greatest_value(self, values):
        return torch.argmax(values, dim=1)

    def __to_percentage(self, correct):
        return 100. * correct / len(self.dataset)

