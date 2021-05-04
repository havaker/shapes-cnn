import torch
import numpy as np
import random
import argparse

from counting import CountingNet, Counting
from counting135 import CountingNet135, Counting135
from classification import ClassificationNet, Classification

def total_number_of_weights(model):
    return sum([val.numel() for key, val in model.state_dict().items()])

def main():
    parser = argparse.ArgumentParser(description="Assignment 1")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--skip", action="store_true", default=False,
                        help="disables training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="plots training")
    parser.add_argument("--confusion-matrix", action="store_true", default=False,
                        help="plots confusion matrix")
    parser.add_argument("--network", type=str, default="classification", metavar="CLASS",
                        help="classification counting or counting135")
    parser.add_argument("--load", type=str, metavar="PATH",
                        help="model file load path")
    parser.add_argument("--save", type=str, metavar="PATH",
                        help="model file save path")
    parser.add_argument("--loss135", action="store_true", default=False,
                        help="computes counting135 train loss")
    parser.add_argument("--slow", action="store_true", default=False,
                        help="train counting135 using slow loss")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    if args.network == "classification":
        net = ClassificationNet()
    elif args.network == "counting":
        net = CountingNet()
    elif args.network == "counting135":
        net = CountingNet135()
    else:
        print("Unknown network type")
        return
    print("Created", args.network, "network")

    print("Total number of weights:", total_number_of_weights(net))

    if args.load:
        print("Loading model from", args.load)
        net.load_state_dict(torch.load(args.load))

    net = net.to(device)

    if args.network == "classification":
        problem = Classification(net, "data/extracted/", device)
    elif args.network == "counting":
        problem = Counting(net, "data/extracted/", device)
    elif args.network == "counting135":
        problem = Counting135(net, "data/extracted/", device, fast_loss = not args.slow)

    if not args.skip:
        print("Training started")
        problem.train(args.epochs)
        print("Training ended")
        if args.plot:
            problem.trainer.plot()
    else:
        print("Skipping training")

    if args.loss135:
        print("Calculating test loss by mapping output tensors to size 60")
        print("    and feeding them to functions computing loss/accuracy")
        print("    from the second network (Counting)")
        avg_loss, avg_accuracy = problem.compute_loss()
        print("Average loss: {:.4f}, Accuracy: {:.2f}%".format(
            avg_loss, avg_accuracy * 100
        ))

    if args.confusion_matrix:
        print("Generating confusion matrix")
        matrix, labels = problem.confusion_matrix(device)
        problem.plot_confusion_matrix(matrix, labels)

    if args.save:
        print("Saving model to", args.save)
        torch.save(net.state_dict(), args.save)

if __name__ == "__main__":
    main()
