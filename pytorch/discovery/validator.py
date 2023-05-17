class PytorchValidator:
    def __init__(self, measurer):
        self.measurer = measurer

    def validate(self, model, validation_dataset):
        measure = 0.
        for entry in validation_dataset:
            inputs, labels = entry.get_input(), entry.get_output()
            predictions = self.__predict(model, inputs)
            measure += self.measurer.measure(predictions, labels)
        return self.__average_quality(measure, len(validation_dataset))

    def __predict(self, model, inputs):
        return model(inputs)

    def __average_quality(self, measure, size):
        return measure / size
