class ModelExplorer:
    def __init__(self, wrappers, training_dataset, validation_dataset, test_task):
        self.training_wrappers = wrappers
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_task = test_task

    def explore(self, epochs, path, mode=max):
        self.__select_best_model([self.__compute_quality(epochs, wrapper) for wrapper in self.training_wrappers], mode).save(path)

    def __select_best_model(self, accuracies, function):
        return self.training_wrappers[accuracies.index(function(accuracies))].get_model()

    def __compute_quality(self, epochs, training_task):
        return self.test_task.execute(training_task.execute(epochs, self.training_dataset, self.validation_dataset))