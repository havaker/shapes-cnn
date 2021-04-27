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
    def __init__(self, hidden=6):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, 6)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = self.fc5(x)
        x = F.sigmoid(x)
        return x

model = Net()

print(type(trainset[0][0]))
print(trainset[0])
print('total number of weights =', total_number_of_weights(model))


def nll_loss_sum(outputs, labels):
    loss = 0
    for i in range(6):
        loss += F.nll_loss(outputs, labels.T[i])
        print('nll loss', i, ' loss', loss)
    return loss

criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_and_evaluate(model, criterion, optimizer, epochs_count=10):
    model.train()
    for epoch in range(epochs_count):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            print("out", outputs[0])
            print("labels", labels[0])
            loss = criterion(outputs, labels.float())
            print("loss", loss)
            print()
            print()

            loss.backward()
            optimizer.step()
    return model

train_and_evaluate(model, criterion, optimizer)

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
