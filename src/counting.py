import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
from itertools import combinations

from shapes import *
from trainer import Trainer

class CountingNet(nn.Module):
    def __init__(self):
        super(CountingNet, self).__init__()

        chan = 4

        self.cnn = nn.Sequential(
            nn.Conv2d(1, chan, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan, chan*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan*2, chan*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 14x14

            nn.Conv2d(chan*4, chan*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan*8, chan*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*16),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan*16, chan*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 7x7

            nn.Conv2d(chan*32, chan*64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*64),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan*64, chan*128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*128),
            nn.ReLU(inplace=True),

            nn.Conv2d(chan*128, chan*256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan*256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, padding=1),
            # 3x3
        )

        self.linear =  nn.Sequential(
            nn.Linear(chan * 256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 60),
            nn.BatchNorm1d(60),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(-1, 6, 10)
        x = F.softmax(x, dim=2)
        return x

class Counting:
    def __init__(self, net, dataset_path, device, is135=False):
        self.net = net
        self.dataset_path = dataset_path

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        test_transforms = transforms.Compose([
            base_transforms,
        ])
        train_transforms = transforms.Compose([
            base_transforms,
            random_flip,
            random_rotate,
        ])

        self.trainset = ShapesDataset(
            self.dataset_path,
            train=True,
            transform=train_transforms
        )
        self.testset = ShapesDataset(
            self.dataset_path,
            train=False,
            transform=test_transforms
        )

        batch_size = 256
        workers_count = 4

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )


        self.device = device
        self.trainer = self.create_trainer(self.criterion_uptempo)
        self.j = torch.arange(0, 10).repeat(6, 1).to(device)

    def criterion_slow(self, outputs, targets):
        loss = 0
        for i in range(6):
            for j in range(10):
                for output, target in zip(outputs, targets):
                    loss += output[i][j] * (j - target[i])**2
        return loss

    def criterion_fast(self, outputs, targets):
        loss = 0
        for output, target in zip(outputs, targets):
            counts = target.repeat(10, 1).T
            d = torch.pow(counts - self.j, 2)
            loss += torch.sum(output * d)
        return loss

    def criterion_uptempo(self, outputs, targets):
        loss = 0
        batch_size = outputs.size(0)

        # if we omit the batch dimension, r will hold a 6x10 matrix, in which
        # each row consists of 10 equal numbers (`r` from given loss formula)
        r = targets.repeat(1, 10).view(batch_size, 10, 6)
        r = r.permute(0, 2, 1)

        j = self.j.repeat(outputs.size(0), 1, 1)

        # diff_sq corresponds (j - r^i)^2 from given loss formula
        diff_sq = torch.pow(r - j, 2)

        loss += torch.sum(outputs * diff_sq)
        return loss

    def encode_count(label):
        n = torch.zeros(6, 10)
        for i, count in enumerate(label):
            n[i][count] = 1
        return n

    def correctly_predicted_count(self, predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            predicted = torch.argmax(predicted, axis=1)
            if torch.allclose(predicted.float(), correct.float()):
                acc += 1
        return acc

    def create_trainer(self, criterion):
        return Trainer(
            model=self.net,
            criterion=criterion,
            correct=self.correctly_predicted_count,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            device=self.device
        )

    def train(self, epochs):
        return self.trainer.train(epochs, 10)
