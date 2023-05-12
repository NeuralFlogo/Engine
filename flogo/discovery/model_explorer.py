class ModelExplorer:
    def __init__(self, training_tasks, test_task):
        self.training_tasks = training_tasks
        self.test_task = test_task

    def explore(self):
        return self.__select_best_model([self.__compute_accuracy(training) for training in self.training_tasks])

    def __select_best_model(self, accuracies):
        return self.training_tasks[accuracies.index(max(accuracies))].architecture, max(accuracies)

    def __compute_accuracy(self, training):
        return self.test_task.test(training.execute())
