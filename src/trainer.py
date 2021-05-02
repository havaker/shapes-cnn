import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(
        self,
        model,
        criterion,
        correct,
        optimizer,
        train_loader,
        test_loader,
        device,
    ):
        self.model = model
        self.criterion = criterion
        self.correct = correct
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train_epoch(self):
        loss_sum = 0
        correct_count = 0

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()

            correct_count += self.correct(output, target)
            loss_sum += loss.item()

        avg_loss = loss_sum / len(self.train_loader.dataset)
        avg_correct = correct_count / len(self.train_loader.dataset)
        return avg_loss, avg_correct

    def test_epoch(self):
        self.model.eval()
        test_loss = 0
        correct_count = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += self.criterion(output, target.float()).item()  # sum up batch loss
                correct_count += self.correct(output, target)

        avg_loss = test_loss / len(self.test_loader.dataset)
        avg_correct = correct_count / len(self.test_loader.dataset)
        return avg_loss, avg_correct

    def train(self, epoch_count, early_stop=None, verbose=True):
        self.avg_train_losses = []
        self.avg_test_losses = []
        self.avg_train_accuracies = []
        self.avg_test_accuracies = []

        for epoch in range(epoch_count):
            if verbose:
                print("Epoch: ", epoch)

            avg_train_loss, avg_train_correct = self.train_epoch()
            if verbose:
                print('    Train set: Average loss: {:.4f}, Accuracy: ({:.0f}%)'.format( 
                    avg_train_loss, 100*avg_train_correct
                ))

            avg_test_loss, avg_test_correct = self.test_epoch()
            if verbose:
                print('    Test set: Average loss: {:.4f}, Accuracy: ({:.0f}%)'.format( 
                    avg_test_loss, 100*avg_test_correct
                ))

            self.avg_train_losses.append(avg_train_loss)
            self.avg_test_losses.append(avg_test_loss)
            self.avg_test_accuracies.append(avg_test_correct)
            self.avg_train_accuracies.append(avg_train_correct)

            if early_stop and len(self.avg_train_losses) > early_stop:
                old_best_loss = min(self.avg_train_losses[:-early_stop])
                last_loss = min(self.avg_train_losses[-early_stop:])
                if last_loss >= old_best_loss:
                    if verbose:
                        print("Early stop")
                    break

        return self.avg_train_losses, self.avg_train_accuracies, \
            self.avg_test_losses, self.avg_test_accuracies

    def plot(self):
        """Plot the training history of the classification model."""
        fig, ax = plt.subplots(1,2, figsize=(20,6), sharex=True)
        x = np.arange(len(self.avg_train_losses))

        ax[0].plot(x, self.avg_train_losses,'b', marker='.', label="average train loss")
        ax[0].plot(x, self.avg_test_losses,'r', marker='.', label="average test loss")
        ax[0].legend()

        ax[1].plot(x, self.avg_train_accuracies,'b', marker='.', label="average train accuracy")
        ax[1].plot(x, self.avg_test_accuracies,'r', marker='.', label="average test accuracy")
        ax[1].legend()
