from torchvision import transforms


def preprocess_images(size, mean=[0, 0, 0], std=[1, 1, 1], horizontal=0, vertical=0, degrees=0, gray=0):
    transformations = [transforms.ToTensor(),
                       transforms.Resize(size=(size, size)),
                       transforms.Normalize(mean, std)]
    # transforms.RandomHorizontalFlip(horizontal),
    # transforms.RandomVerticalFlip(vertical),
    # transforms.RandomRotation(degrees),
    # transforms.RandomGrayscale(gray)]
    return transforms.Compose(transformations)  # TODO crop
