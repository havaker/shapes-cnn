import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import zipfile
import pandas as pd
import io
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
import torchvision.transforms as transforms

# https://docs.python.org/3/library/zipfile.html#zipfile-objects
f = zipfile.ZipFile('./gsn-2021-1.zip')
labels_csv = io.BytesIO(f.read('data/labels.csv'))
labels_df = pd.read_csv(labels_csv)

print(labels_df.head())

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ShapesDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.f = zipfile.ZipFile(path)
        labels_csv = io.BytesIO(f.read('data/labels.csv'))
        if train:
            self.labels = pd.read_csv(labels_csv).iloc[:9000]
        else:
            self.labels = pd.read_csv(labels_csv).iloc[-1000:]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.labels.iloc[idx]['name']
        labels = self.labels.iloc[idx][1:].values

        img_path_in_zip = 'data/' + img_name
        img_file = io.BytesIO(self.f.read(img_path_in_zip))
        image = imread(img_file)

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

t = transforms.Compose([
    ImageTransform(transforms.Compose([
        transforms.ToTensor(),
        to_monochrome,
        # convert range of tensors from [0, 1] to [-1, 1]
        transforms.Normalize((0.5), (0.5)), 
    ])),
    #LabelTransform(to_classification),
    random_flip,
    random_rotate
]);

shapes_dataset = ShapesDataset('./gsn-2021-1.zip', train=True, transform=t)
img, lab = shapes_dataset[0]
print(img.size())
print(img)

'''
fig = plt.figure()

for i in range(len(shapes_dataset)):
    image, labels = shapes_dataset[i]
    print(labels)

    image = image.numpy().transpose((1, 2, 0))
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(image)

    if i == 3:
        plt.show()
        break

# możee
# https://jbencook.com/torchvision-transforms/
'''
