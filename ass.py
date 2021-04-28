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
from dataset import *

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

base_transforms =  transforms.Compose([
    ImageTransform(transforms.Compose([
        transforms.ToTensor(),
        to_monochrome,
        # convert range of tensors from [0, 1] to [-1, 1]
        transforms.Normalize((0.5), (0.5)), 
    ])),
    LabelTransform(transforms.Compose([
        #torch.from_numpy
    ])),
]);

def total_number_of_weights(model):
    return sum([val.numel() for key, val in model.state_dict().items()])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.fc = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 6)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.15, inplace=True)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net()
#model.load_state_dict(torch.load("model-69.torch"))
modul = model.to(device)

print('total number of weights =', total_number_of_weights(model))

class Trainer():
    def __init__(self, model, criterion, correct, optimizer, train_loader, test_loader):
        self.model = model
        self.criterion = criterion
        self.correct = correct
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_epoch(self):
        loss_sum = 0
        samples_count = 0

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(self.train_loader.dataset)


    def test_epoch(self):
        self.model.eval()
        test_loss = 0
        correct_count = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)

                test_loss += self.criterion(output, target.float()).item()  # sum up batch loss
                correct_count += self.correct(output, target)

        avg_loss = test_loss / len(self.test_loader.dataset)
        avg_correct = correct_count / len(self.test_loader.dataset)
        return avg_loss, avg_correct

    def train(self, epoch_count):
        for epoch in range(epoch_count):
            print("Epoch: ", epoch)

            avg_train_loss = self.train_epoch()
            print("    Train set: Average loss: {:.4f}".format(avg_train_loss))

            avg_test_loss, avg_test_correct = self.test_epoch()
            print('    Test set: Average loss: {:.4f}, Accuracy: ({:.0f}%)'.format( 
                avg_test_loss, 100*avg_test_correct
            ))

class Classification:
    def __init__(self, net, dataset_path):
        self.net = net
        self.dataset_path = dataset_path

        self.criterion = nn.BCELoss(reduction='sum')
        #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
        self.optimizer = optim.Adam(model.parameters())

        self.test_transforms = transforms.Compose([
            base_transforms,
            LabelTransform(to_classification),
        ]);

        self.train_transforms = transforms.Compose([
            self.test_transforms,
            random_flip,
            random_rotate
        ])

    def correctly_predicted_count(predicted_labels_batch, correct_labels_batch):
        acc = 0
        for i, (predicted, correct) in enumerate(zip(predicted_labels_batch, correct_labels_batch)):
            two_highest = torch.sort(predicted).indices[-2:]
            acc += 1 if torch.sum(correct[two_highest]).item() == 2 else 0
        return acc


    def create_trainer(self, batch_size=512, workers_count=4):
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
            #correct=None,
            optimizer=self.optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
        )

    def train(self, epochs):
        self.trainer.train(epochs)


c = Classification(model, 'gsn-2021-1.zip')
c.create_trainer()
c.train(100)

#print("saving model")
#torch.save(model.state_dict(), 'model-69.torch')
#print("saved")
#test(model, criterion, testloader, correctly_predicted_count)

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
