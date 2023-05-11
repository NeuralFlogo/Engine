class Entry:

    def __init__(self, size, inputs, outputs):
        self.size = size
        self.inputs = inputs
        self.outputs = outputs

    def get_size(self):
        return self.size

    def get_input(self):
        return self.inputs

    def get_output(self):
        return self.outputs
