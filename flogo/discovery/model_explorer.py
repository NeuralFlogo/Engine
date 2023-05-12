class ModelExplorer:
    def __init__(self, wrappers, training_dataset, validation_dataset, test_task):
        self.training_wrappers = wrappers
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_task = test_task

    def explore(self, epochs):
        return self.__select_best_model([self.__compute_accuracy(wrapper, epochs) for wrapper in self.training_wrappers])

    def __select_best_model(self, accuracies):
        return self.training_wrappers[accuracies.index(max(accuracies))].get_model(), max(accuracies)

    def __compute_accuracy(self, training_wrapper, epochs):
        return self.test_task.execute(training_wrapper.execute(epochs, self.training_dataset, self.validation_dataset))
