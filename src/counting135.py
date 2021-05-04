import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
from itertools import combinations

from shapes import *
from trainer import Trainer

class CountingNet135(nn.Module):
    def __init__(self):
        super(CountingNet135, self).__init__()

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
            nn.MaxPool2d(kernel_size=2),
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
            nn.MaxPool2d(kernel_size=2),
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

            nn.Linear(256, 15*9),
            #nn.BatchNorm1d(15*9),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        #x = F.softmax(x, dim=1)
        x = x.view(x.size(0), 15, 9)
        return x

class Counting135:
    def __init__(self, net, dataset_path, device, fast_loss=True):
        self.net = net
        self.dataset_path = dataset_path
        self.fast_loss = fast_loss

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.trainset, self.train_loader = self.create_dataset_and_loader(
            batch_size=256,
            workers_count=4,
            train=True,
            encode_label_to_135 = fast_loss
        )
        self.testset, self.test_loader = self.create_dataset_and_loader(
            batch_size=256,
            workers_count=4,
            train=False,
            encode_label_to_135 = fast_loss
        )

        self.device = device
        self.trainer = self.create_trainer(self.criterion)
        self.j = torch.arange(0, 10).repeat(6, 1).to(device)

    # outputs.size() == [batch_size, 135]
    # targets.size() == [batch_size, 135 if self.fast_loss else 6]
    def criterion(self, outputs, targets):
        if self.fast_loss:
            targets = torch.argmax(targets.view(targets.size(0), 15 * 9) , dim=1)
            outputs = outputs.view(outputs.size(0), 15 * 9)
            return F.cross_entropy(outputs, targets)
        else:
            return self.criterion135to60(outputs, targets)

    def criterion135to60(self, outputs, targets):
        #encoded_label = self.encode_label_to_135(targets[0])
        outputs = F.softmax(outputs.view(outputs.size(0), -1), dim=1).view(outputs.size(0), 15, 9)
        o60 = Counting135.encode_60_from_135(outputs)
        #print("targets =", targets)
        #print("outputs =", outputs)
        #print("encoded =", encoded_label)
        #print("o60 =", o60)
        return self.criterion60(o60, targets)


    def encode_60_from_135(batch):
        result = torch.zeros(batch.size(0), 6, 10).to(batch.device)

        for batch_number, x in enumerate(batch):
            sum_of_c = torch.zeros(6).to(batch.device)
            x = x.view(15, 9)
            for i, (c1, c2) in enumerate(combinations(range(6), 2)):
                for div in range(1, 10):
                    c1_count = div
                    c2_count = 10 - div

                    sum_of_c[c1] += x[i][div-1]
                    sum_of_c[c2] += x[i][div-1]

                    result[batch_number][c1][c1_count] += x[i][div-1]
                    result[batch_number][c2][c2_count] += x[i][div-1]

            sum_of_x = torch.sum(x)
            for i in range(6):
                result[batch_number][i][0] = sum_of_x - sum_of_c[i]

        return result


    # outputs.size() == [batch_size, 60]
    # targets.size() == [batch_size, 6]
    def criterion60(self, outputs, targets):
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

    def loss135(self):
        testset, test_loader = self.create_dataset_and_loader(
            batch_size=100,
            workers_count=4,
            train=False,
            encode_label_to_135=False
        )

        self.net.eval()
        test_loss = 0
        correct_count = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)

                test_loss += self.criterion135to60(output, target.float()).item()  # sum up batch loss
                #correct_count += 1
                if correct_count > 2:
                    break
                #correct_count += self.correct(output, target)

        avg_loss = test_loss / len(test_loader.dataset)
        avg_correct = correct_count / len(test_loader.dataset)
        return avg_loss, avg_correct

    def encode_label_to_135(self, label):
        n = torch.zeros(15, 9)
        for i, (c1, c2) in enumerate(combinations(range(6), 2)):
            for div in range(1, 10):
                n[i][div-1] = 1 if label[c1] == div and label[c2] == 10 - div else 0
        return n.float()

    def encode_count(label):
        n = torch.zeros(6, 10)
        for i, count in enumerate(label):
            n[i][count] = 1
        return n

    def correctly_predicted_count(self, predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            predicted = torch.argmax(predicted.view(15*9)).item()
            if correct_labels_batch.size(1) == 15:
                correct = torch.argmax(correct.view(15*9)).item()
            else:
                correct = torch.argmax(self.encode_label_to_135(correct).view(15*9)).item()
            if predicted == correct:
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

    def create_dataset_and_loader(
            self,
            batch_size,
            workers_count,
            train=True,
            encode_label_to_135=True
        ):

        transforms_list = [base_transforms]
        if train:
            transforms_list.extend([random_flip, random_rotate])
        if encode_label_to_135:
            transforms_list.append(LabelTransform(self.encode_label_to_135))

        composed = transforms.Compose(transforms_list)

        dataset = ShapesDataset(
            self.dataset_path,
            train,
            composed
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=workers_count
        )

        return dataset, loader
