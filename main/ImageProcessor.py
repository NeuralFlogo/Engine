from torchvision import transforms


def preprocess_images(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


def preprocess_images(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])


def preprocess_images():
    return transforms.Compose([transforms.ToTensor()])
