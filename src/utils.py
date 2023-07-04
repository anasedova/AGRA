import torch
import numpy as np
import random

from src.f1loss import F1Loss, F1LossMulti


def set_seed(seed: int) -> None:
    """fix seed for all shuffle processes in order to get the reproducible result"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_weights(train_labels) -> np.array:
    """computes sample weights for comparison batch if class-weighted sampling is applied"""

    # determine the number of classes and samples
    num_classes = len(np.unique(train_labels))
    num_samples = len(train_labels)

    # count occurrences per label
    _, counts = np.unique(train_labels, return_counts=True)

    assert sum(counts) == num_samples

    # initialize weights, source/cite
    weights = np.zeros(num_samples)
    for label in range(0, num_classes):
        weights[train_labels == label] = 1 / counts[label]

    return weights


def get_loss(comp_loss: str, num_classes: int, multilabel: bool):
    """retrieve the loss function"""
    if comp_loss == 'F1':
        if multilabel is False:
            comp_loss = F1Loss(num_classes)
        else:
            comp_loss = F1LossMulti()
        loss_type = 'sum'
    else:
        if multilabel is False:
            comp_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            comp_loss = torch.nn.BCELoss(reduction='mean')
        loss_type = 'mean'
    return comp_loss, loss_type

