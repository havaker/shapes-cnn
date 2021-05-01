import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.metrics import confusion_matrix, classification_report

import random
import itertools

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

        self.criterion = nn.BCELoss(reduction="sum")
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

        self.device = device
        self.create_trainer()

    def correctly_predicted_count(predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            two_highest = torch.sort(predicted).indices[-2:]
            acc += 1 if torch.sum(correct[two_highest]).item() == 2 else 0
        return acc

    def confusion_matrix(self, device):
        disambiguate_pair = lambda p: (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
        label_pairs = lambda labels: map(disambiguate_pair, itertools.combinations(labels, 2))

        all_unordered_pairs = list(label_pairs([0, 1, 2, 3, 4, 5]))
        pairs_to_index = {}
        for i, pair in enumerate(all_unordered_pairs):
            pairs_to_index[pair] = i
        classes_count = len(all_unordered_pairs)

        matrix = torch.zeros(classes_count, classes_count)

        p = np.zeros(len(self.test_loader.dataset))
        t = np.zeros(len(p))

        self.net.eval()
        with torch.no_grad():
            i = 0
            for data_batch, target_batch in self.test_loader:
                data_batch = data_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                output_batch = self.net(data_batch)

                for output, target in zip(output_batch, target_batch):
                    predicted_labels = torch.sort(output).indices[-2:].tolist()
                    true_labels = (target == 1).nonzero(as_tuple=True)[0].tolist()

                    predicted_labels = disambiguate_pair(tuple(predicted_labels))
                    true_labels = disambiguate_pair(tuple(true_labels))
                    
                    predicted_labels_index = pairs_to_index[predicted_labels]
                    true_labels_index = pairs_to_index[true_labels]

                    matrix[predicted_labels_index][true_labels_index] += 1
                    p[i] = predicted_labels_index
                    t[i] = true_labels_index
                    i += 1
        
        m = confusion_matrix(t, p)
        return m, all_unordered_pairs

    # src: https://www.kaggle.com/fuzzywizard/fashion-mnist-cnn-keras-accuracy-93#6)-Confusion-Matrix
    def plot_confusion_matrix(
        self,
        cm,
        classes,
        normalize=False,
        title="Confusion matrix",
        cmap=plt.cm.Blues
    ):
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def create_trainer(self, batch_size=512, workers_count=4):
        self.train_set = ShapesDataset(
            self.dataset_path,
            train=True,
            transform=self.train_transforms
        )
        self.test_set = ShapesDataset(
            self.dataset_path,
            train=False,
            transform=self.test_transforms
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_count
        )

        self.trainer = Trainer(
            model=self.net,
            criterion=self.criterion,
            correct=Classification.correctly_predicted_count,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            device=self.device
        )

    def train(self, epochs):
        return self.trainer.train(epochs, 10)

class CountingNet(nn.Module):
    def __init__(self):
        super(CountingNet, self).__init__()

        dropout = 0.2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1), #28x28 27x27

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1), #27x27 26x26

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), #26x26 13x13

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=1), # 13x13 -> 12x12

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2), # 12x12 -> 6x6

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=1), # 6x6 -> 3x3
        )

        self.linear =  nn.Sequential(
            nn.Linear(512 * 5 * 5, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(128),

            nn.Linear(128, 60),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(-1, 6, 10)
        x = F.softmax(x, dim=2)
        x = x.view(x.size(0), -1)
        return x

class Counting:
    def __init__(self, net, dataset_path, device):
        self.net = net
        self.dataset_path = dataset_path

        #self.criterion = nn.MSELoss(reduction="sum")
        self.criterion = self.criterion_ultra
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
            print("\n i =", i)
            ex_label = ex_label.view(6,10)
            out = net(ex_img.view(1, 1, 28, 28).to(device)).view(6,10)
            print("label =", ex_label)
            print("out =", out)
            predicted = torch.argmax(out, axis=1)
            correct = torch.argmax(ex_label, axis=1)
            print("predicted =", predicted)
            print("correct =", correct)

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clnet = ClassificationNet()
    clnet.load_state_dict(torch.load("models/classification.model"))
    clnet = clnet.to(device)


    print("total number of weights =", total_number_of_weights(clnet))

    c = Classification(clnet, "data/extracted/", device)
    cm, labels = c.confusion_matrix(device)
    labels = list(map(lambda l: str(l), labels))
    print(cm)
    print(labels)
    c.plot_confusion_matrix(cm, labels)
    return
    c.train(300)

    print("saving model")
    torch.save(model.state_dict(), "models/recent.model")
    print("saved")


    conet = CountingNet()
    #conet.load_state_dict(torch.load("models/recent.model"))
    conet = conet.to(device)

    counting = Counting(conet, "data/extracted/", device)
    counting.train(300)
    print("total number of weights =", total_number_of_weights(conet))

    torch.save(conet.state_dict(), "models/recent.model")

    return

if __name__ == "__main__":
    main()
