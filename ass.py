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

from shapes import *
from trainer import Trainer

def total_number_of_weights(model):
    return sum([val.numel() for key, val in model.state_dict().items()])

class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()

        dropout = 0.2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2), # 32x28x28 -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), # 64x14x14 -> 64x7x7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), # 128x7x7 -> 128x3x3
        )

        self.linear =  nn.Sequential(
            nn.Linear(128 * 3 * 3, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(32, 6),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Classification:
    def __init__(self, net, dataset_path, device):
        self.net = net
        self.dataset_path = dataset_path

        self.criterion = nn.BCELoss(reduction='sum')
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.test_transforms = transforms.Compose([
            base_transforms,
            LabelTransform(to_classification),
        ]);

        self.train_transforms = transforms.Compose([
            self.test_transforms,
            random_flip,
            random_rotate
        ])

        self.create_trainer(device)

    def correctly_predicted_count(predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            two_highest = torch.sort(predicted).indices[-2:]
            acc += 1 if torch.sum(correct[two_highest]).item() == 2 else 0
        return acc

    def create_trainer(self, device, batch_size=512, workers_count=4):
        trainset = ShapesDataset(
            self.dataset_path,
            train=True,
            transform=self.train_transforms
        )
        testset = ShapesDataset(
            self.dataset_path,
            train=False,
            transform=self.test_transforms
        )

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )

        self.trainer = Trainer(
            model=self.net,
            criterion=self.criterion,
            correct=Classification.correctly_predicted_count,
            optimizer=self.optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

    def train(self, epochs):
        return self.trainer.train(epochs, 10)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnet = ClassificationNet()
    cnet = cnet.to(device)

    print("total number of weights =", total_number_of_weights(cnet))

    c = Classification(cnet, "extracted/", device)
    c.train(300)

    print("saving model")
    torch.save(model.state_dict(), "model-c.torch")
    print("saved")

if __name__ == '__main__':
    main()

'''
model.eval()
ex_img, ex_label = testset[1]
predicted_label = model(ex_img.view(1, *ex_img.size()).to(device))
print("true label =", ex_label)
print("predicted label =", predicted_label)

print("hej")
total_loss = 0
for images, labels in trainloader:
    total_loss += len(images) + len(labels)
print("juz po")

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

print("begin")
#X = []
#y = []
#for images, labels in trainloader:
    #X.extend(images)
    #y.extend(labels)
print("end")

'''
