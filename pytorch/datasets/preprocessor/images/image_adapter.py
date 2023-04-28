from torchvision import transforms


class ImageAdapter:

    def __init__(self, size, mean, std, horizontal=0, vertical=0, degrees=0, gray=0):
        self.size = size
        self.mean = mean
        self.std = std
        self.horizontal = horizontal
        self.vertical = vertical
        self.degrees = degrees
        self.gray = gray

    def adaptions(self):
        transformations = [transforms.ToTensor(),
                           transforms.Resize(size=(self.size, self.size)),
                           transforms.Normalize(self.mean, self.std)]
        # transforms.RandomHorizontalFlip(self.horizontal),
        # transforms.RandomVerticalFlip(self.vertical),
        # transforms.RandomRotation(self.degrees),
        # transforms.RandomGrayscale(self.gray)]
        return transforms.Compose(transformations)  # TODO crop
