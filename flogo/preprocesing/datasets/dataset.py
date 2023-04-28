class Dataset:

    def __init__(self, inputs, outputs, mapper, batch_size):
        self.inputs = inputs
        self.outputs = outputs
        self.mapper = mapper
        self.batch_size = batch_size

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        output = self.outputs[idx]
        return self.mapper.map(inputs), self.mapper.map(output)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        inputs, output = [], []
        counter = 0
        while self.index < len(self) and counter < self.batch_size:
            inputs.append(self.inputs[self.index])
            output.append(self.outputs[self.index])
            self.index += 1
            counter += 1
        return self.mapper.map(inputs), self.mapper.map(output)

    def __get_positions(self, test_proportion, validation_proportion):
        validation_position = round(len(self) - (len(self) * validation_proportion))
        test_position = round(validation_position - (len(self) * test_proportion))
        return test_position, validation_position

    def divide_to(self, test_proportion=0, validation_proportion=0):
        test_position, validation_position = self.__get_positions(test_proportion, validation_proportion)
        train_dataset = Dataset(self.inputs[:test_position],
                                self.outputs[:test_position],
                                self.mapper,
                                self.batch_size)
        test_dataset = Dataset(self.inputs[test_position:validation_position],
                               self.outputs[test_position:validation_position],
                               self.mapper,
                               self.batch_size)
        validation_dataset = Dataset(self.inputs[validation_position:],
                                     self.outputs[validation_position:],
                                     self.mapper,
                                     self.batch_size)
        return train_dataset, test_dataset, validation_dataset

    @staticmethod
    def get(transformer, mapper, batch_size):
        return Dataset(transformer.transform_inputs(),
                       transformer.transform_outputs(),
                       mapper,
                       batch_size)
