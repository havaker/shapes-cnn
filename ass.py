import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.io import imread

import random
import zipfile
import io

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
        labels = self.labels.iloc[idx][1:].values.astype(int)

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
    LabelTransform(transforms.Compose([
        to_classification,
        torch.from_numpy
    ])),
    #random_flip,
    #random_rotate
]);

trainset = ShapesDataset('./gsn-2021-1.zip', train=True, transform=t)
testset = ShapesDataset('./gsn-2021-1.zip', train=False, transform=t)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=1
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=True,
    num_workers=1
)

def total_number_of_weights(model):
    return sum([val.numel() for key, val in model.state_dict().items()])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.fc = nn.Linear(16 * 3 * 3, 6)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

model = Net()

print('total number of weights =', total_number_of_weights(model))

criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(model, criterion, optimizer, epochs_count=10):
    model.train()
    for epoch in range(epochs_count):
        total_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainset)
        print('avg loss =', avg_loss)

    return model

train_and_evaluate(model, criterion, optimizer)

ex_img, ex_label = trainset[0]
predicted_label = model(ex_img.view(1, *ex_img.size()))
print("true label =", ex_label)
print("predicted label =", predicted_label)

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
