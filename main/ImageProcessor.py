from torchvision import transforms
from random import randint


def preprocess_images(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])


def preprocess_images(size, probability):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(probability),
        transforms.RandomVerticalFlip(probability)])


def preprocess_images(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


def preprocess_images(size, mean, std, probability):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(probability),
        transforms.RandomVerticalFlip(probability)])


def preprocess_images(size, mean, std, probability, degrees):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(probability),
        transforms.RandomVerticalFlip(probability),
        transforms.RandomRotation(degrees)])


def preprocess_images(size, mean, std, probability, degrees):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(probability),
        transforms.RandomVerticalFlip(probability),
        transforms.RandomRotation(degrees),
        transforms.RandomCrop(size=(size / randint(0, size), size / randint(0, size)))])
