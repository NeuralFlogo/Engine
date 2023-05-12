class ModelExplorer:
    def __init__(self, models, training_dataset, validation_dataset, training_tasks, test_task):
        self.models = models
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.training_tasks = training_tasks
        self.test_task = test_task

    def explore(self, epochs):
        return self.__select_best_model([self.__compute_accuracy(index, epochs) for index in range(self.training_tasks)])

    def __select_best_model(self, accuracies):
        return self.models[accuracies.index(max(accuracies))], max(accuracies)

    def __compute_accuracy(self, index, epochs):
        return self.test_task.execute(
            self.training_tasks[index].execute(epochs, self.models[index], self.training_dataset, self.validation_dataset)
        )
