from compiled.model.layers.flatten import Flatten


class CompiledFlattenBlock:
    def __init__(self, flatten: Flatten):
        self.flatten = flatten
