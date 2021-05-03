import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import itertools

from shapes import *
from trainer import Trainer

class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()

        dropout = 0.2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 32x28x28 -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2), # 64x14x14 -> 64x7x7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
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

        all_pairs_str = map(
            lambda pair: label_names[pair[0]] + "+" + label_names[pair[1]],
            all_unordered_pairs
        )

        pred_y = np.zeros(len(self.test_loader.dataset))
        true_y = np.zeros(len(pred_y))

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

                    pred_y[i] = predicted_labels_index
                    true_y[i] = true_labels_index
                    i += 1
        
        matrix = confusion_matrix(true_y, pred_y)
        return matrix, list(all_pairs_str)

    # https://www.kaggle.com/fuzzywizard/fashion-mnist-cnn-keras-accuracy-93#6)-Confusion-Matrix
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

