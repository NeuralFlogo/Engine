from torchvision import transforms


def preprocess_images(size, mean, std, horizontal=0, vertical=0, degrees=0, crop=(0, 0), gray=0):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(horizontal),
        transforms.RandomVerticalFlip(vertical),
        transforms.RandomRotation(degrees),
        transforms.RandomCrop(crop),
        transforms.RandomGrayscale(gray)])
