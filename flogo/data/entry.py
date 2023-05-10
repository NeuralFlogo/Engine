class Entry:

    def __init__(self, inputs, outputs, size):
        self.inputs = inputs
        self.outputs = outputs
        self.size = size

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_size(self):
        return self.size
