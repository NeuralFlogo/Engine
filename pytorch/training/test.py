import torch


class Testing:

    def __init__(self, model, testing_loader):
        self.model = model
        self.testing_loader = testing_loader

    def test(self):
        correct = 0.
        with torch.no_grad():
            for i, data in enumerate(self.testing_loader):
                inputs, labels = data
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                expected = torch.argmax(labels, dim=1)
                correct += torch.sum(torch.eq(predictions, expected)).item()
        print('\nTest: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(self.testing_loader.dataset),
            100. * correct / len(self.testing_loader.dataset)))
