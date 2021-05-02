import torch

from counting import CountingNet, Counting
from classification import ClassificationNet, Classification

def total_number_of_weights(model):
    return sum([val.numel() for key, val in model.state_dict().items()])

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device( "cpu")

    clnet = ClassificationNet()
    clnet.load_state_dict(torch.load("models/classification.model"))
    clnet = clnet.to(device)

    conet = CountingNet(clnet)
    conet.load_state_dict(torch.load("models/recent.model"))
    conet = conet.to(device)

    counting = Counting(conet, "data/extracted/", device)
    counting.train(300)
    print("total number of weights =", total_number_of_weights(conet))

    torch.save(conet.state_dict(), "models/recent.model")

    print("total number of weights =", total_number_of_weights(clnet))

    #c = Classification(clnet, "data/extracted/", device)
    #matrix, labels = c.confusion_matrix(device)
    #c.plot_confusion_matrix(matrix, labels)

    #c.train(300)

    return

    #print("saving model")
    #torch.save(model.state_dict(), "models/recent.model")
    #print("saved")



    return

if __name__ == "__main__":
    main()
