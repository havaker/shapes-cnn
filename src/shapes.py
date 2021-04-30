import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.io import imread


class ShapesDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform
        self.path = path

        labels_csv_path = path + "/labels.csv"
        if train:
            self.labels = pd.read_csv(labels_csv_path).iloc[:9000]
        else:
            self.labels = pd.read_csv(labels_csv_path).iloc[-1000:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.labels.iloc[idx]["name"]
        labels = self.labels.iloc[idx][1:].values.astype(int)

        img_path = self.path + "/" + img_name
        image = imread(img_path)

        sample = (image, labels)
        if self.transform:
            sample = self.transform(sample)
        return sample

def to_classification(labels):
    """Convert labels, so they can be used in shape classification."""
    return np.minimum(labels, np.ones(labels.size, dtype=np.int32))

def to_monochrome(image):
    image = torch.mean(image[:-1], axis=0)
    return image.view(1, *image.size())

def random_rotate(sample):
    """Random rotate image and update labels."""
    image, labels = sample

    up = 2
    right = 3
    down = 4
    left = 5

    rotation = random.choice([0, 90, 180, 270])
    image = transforms.functional.rotate(image, angle=rotation)

    if rotation == 90:
        labels[left], labels[down], labels[right], labels[up] = \
            labels[up], labels[left], labels[down], labels[right]
    elif rotation == 180:
        labels[down], labels[right], labels[up], labels[left] = \
            labels[up], labels[left], labels[down], labels[right]
    elif rotation == 270:
        labels[right], labels[up], labels[left], labels[down] = \
            labels[up], labels[left], labels[down], labels[right]

    return image, labels

def random_flip(sample):
    """Flip image and update labels."""
    image, labels = sample

    up = 2
    right = 3
    down = 4
    left = 5

    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        labels[left], labels[right] = labels[right], labels[left]
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        labels[up], labels[down] = labels[down], labels[up]

    return image, labels

class ImageTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, labels = sample
        return (self.transform(image), labels)

class LabelTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, labels = sample
        return (image, self.transform(labels))

base_transforms = ImageTransform(
    transforms.Compose([
        transforms.ToTensor(),
        to_monochrome,
        # convert range of tensors from [0, 1] to [-1, 1]
        transforms.Normalize((0.5), (0.5)), 
    ])
)
