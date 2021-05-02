import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from shapes import *
from trainer import Trainer

class CountingNet(nn.Module):
    def __init__(self, clnet):
        super(CountingNet, self).__init__()
        self.clnet = clnet

        dropout = 0.2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), #28x28 27x27

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), #26x26 13x13

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), # 12x12 -> 6x6

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=1), # 6x6 -> 3x3
        )

        self.linear =  nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(256),
        )

        self.ending =  nn.Sequential(
            nn.Linear(256 + 6, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 60),
        )
        
    def forward(self, x):
        #self.clnet.train()
        with torch.no_grad():
            cl = self.clnet(x)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.linear(x)
        x = torch.cat([cl, x], dim=1)
        x = self.ending(x)
        x = x.view(-1, 6, 10)
        x = F.softmax(x, dim=2)
        x = x.view(x.size(0), -1)
        return x

class Counting:
    def __init__(self, net, dataset_path, device):
        self.net = net
        self.dataset_path = dataset_path

        #self.criterion = nn.MSELoss(reduction="sum")
        self.criterion = self.criterion_fast
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.test_transforms = transforms.Compose([
            base_transforms,
            LabelTransform(Counting.encode_count),
        ])
        self.train_transforms = transforms.Compose([
            self.test_transforms,
            random_flip,
            random_rotate
        ])

        self.create_trainer(device, batch_size=512)
        self.a = torch.arange(0, 10).repeat(6, 1).to(device)
        
        net.eval()
        for i in range(10):
            ex_img, ex_label = self.trainset[i]
            #print("\n i =", i)
            ex_label = ex_label.view(6,10)
            out = net(ex_img.view(1, 1, 28, 28).to(device)).view(6,10)
            #print("label =", ex_label)
            #print("out =", out)
            predicted = torch.argmax(out, axis=1)
            correct = torch.argmax(ex_label, axis=1)
            #print("predicted =", predicted)
            #print("correct =", correct)

    def criterion_slow(self, outputs, targets):
        loss = 0
        for i in range(6):
            for j in range(10):
                for output, target in zip(outputs, targets):
                    output = output.view(6, 10)
                    target = target.view(6, 10)
                    correct = torch.argmax(target, axis=1)
                    loss += output[i][j] * (j - correct[i])**2
        return loss

    def criterion_fast(self, outputs, targets):
        loss = 0
        for output, target in zip(outputs, targets):
            output = output.view(6, 10)
            target = target.view(6, 10)
            counts = torch.argmax(target, axis=1)
            counts = counts.repeat(10, 1).T
            d = torch.pow(counts - self.a, 2)
            loss += torch.sum(output * d)
        return loss

    def criterion_ultra(self, outputs, targets):
        loss = 0
        outputs = outputs.view(-1, 6, 10)
        targets = targets.view(-1, 6, 10)
        counts = torch.argmax(targets, axis=2)
        counts = counts.repeat(1, 10).view(outputs.size(0), 10, 6)
        counts = counts.permute(0, 2, 1)
        a = self.a.repeat(outputs.size(0), 1, 1)
        d = torch.pow(counts - a, 2)
        loss += torch.sum(outputs * d)
        return loss

    def encode_count(label):
        n = torch.zeros(6, 10)
        for i, count in enumerate(label):
            n[i][count] = 1
        return n.flatten()

    def correctly_predicted_count(predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            predicted = predicted.view(6, 10)
            correct = correct.view(6, 10)
            #print("predicted =", predicted)
            #print("correct =", correct)
            predicted = torch.argmax(predicted, axis=1)
            correct = torch.argmax(correct, axis=1)
            #print("predicted =", predicted)
            #print("correct =", correct)
            if torch.allclose(predicted, correct):
                acc += 1
        return acc

    def create_trainer(self, device, batch_size=512, workers_count=4):
        self.trainset = ShapesDataset(
            self.dataset_path,
            train=True,
            transform=self.train_transforms
        )
        self.testset = ShapesDataset(
            self.dataset_path,
            train=False,
            transform=self.test_transforms
        )

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )

        self.trainer = Trainer(
            model=self.net,
            criterion=self.criterion,
            correct=Counting.correctly_predicted_count,
            optimizer=self.optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

    def train(self, epochs):
        return self.trainer.train(epochs, 10)


