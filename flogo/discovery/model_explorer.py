class ModelExplorer:
    def __init__(self, training_tasks, test_task):
        self.training_tasks = training_tasks
        self.test_task = test_task

    def explore(self):
        return self.find_best_model([self.__compute_accuracy(trainer) for trainer in self.training_tasks])

    def __compute_accuracy(self, trainer):
        return self.test_task.test(trainer.execute())

    def find_best_model(self, accuracies):
        return self.training_tasks[accuracies.index(max(accuracies))].architecture, max(accuracies)
