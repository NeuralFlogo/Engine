from flogo.data.dataset import Dataset
from flogo.discovery.tasks.training_task import TrainingTask


class TrainingWrapper:

    def __init__(self, model, training_task: TrainingTask):
        self.model = model
        self.training_task = training_task

    def get_model(self):
        return self.model

    def execute(self, epochs, train_dataset: Dataset, validation_dataset: Dataset):
        return self.training_task.execute(epochs, self.model, train_dataset, validation_dataset)
