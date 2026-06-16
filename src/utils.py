import torch


def accuracy_fn(outputs, labels):

    _, preds = torch.max(outputs, 1)

    correct = (preds == labels).sum().item()

    return correct