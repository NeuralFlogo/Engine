class TorchFreezer:
    def freeze(self, architecture):
        for param in architecture.parameters():
            param.requires_grad = False
