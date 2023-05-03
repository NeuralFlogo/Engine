class ModelExplorer:
    def __init__(self, training_tasks, test_task):
        self.training_tasks = training_tasks
        self.test_task = test_task

    def explore(self):
        accuracies = []
        for strategy in self.training_tasks:
            accuracies.append(self.__compute_accuracy(strategy))
        return self.find_best_model(accuracies)

    def __compute_accuracy(self, strategy):
        return self.test_task.execute(strategy.implement())

    def find_best_model(self, accuracies):
        return self.training_tasks.index(max(accuracies)).architecture
