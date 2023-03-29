from torchvision import transforms


def preprocess_images(size, mean=None, std=None, horizontal=0, vertical=0, degrees=0, crop=(0, 0), gray=0):
    transformations = [transforms.Resize(size),
                       transforms.RandomHorizontalFlip(horizontal),
                       transforms.RandomVerticalFlip(vertical),
                       transforms.RandomRotation(degrees),
                       transforms.RandomCrop(crop),
                       transforms.RandomGrayscale(gray)]
    if mean:
        transformations.append(transforms.Normalize(mean, std))
    return transforms.Compose(transformations)
