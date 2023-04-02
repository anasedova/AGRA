import torch
import numpy as np
import random

from src.f1loss import F1Loss


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_weights(train_labels) -> np.array:
    """
    computes sample weights for comparison batch
    """

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

    # normalize weights
    # vec_sum = np.sum(weights)
    # weights = weights/vec_sum
    # assert np.sum(weights) == 1

    return weights


# get loss
def get_loss(comp_loss: str, num_classes: int):
    if comp_loss == 'F1':
        comp_loss = F1Loss(num_classes)
    else:
        comp_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    return comp_loss


def get_statistics(
        weak_labels: torch.tensor, chosen_labels: torch.tensor, gold_labels: torch.tensor, other: int,
        ignore_index: int
):
    """
    computes label change statistics
    :param chosen_labels: tensor of weak labels for the batch
    :param chosen_labels: tensor of labels for the batch chosen by AGRA
    :param gold_labels: tensor of gold labels for the batch
    :param other: alternative label
    :return: statistics on the label changes
    """

    # no alternative label
    if other is None:
        # Correctly labeled and kept
        correctly_labeled = torch.where(weak_labels == gold_labels)[0]
        kept = torch.where(chosen_labels != ignore_index)[0]
        correctly_kept = len(np.intersect1d(correctly_labeled, kept))

        # Correctly labeled and removed
        removed = torch.where(chosen_labels == ignore_index)[0]
        falsely_removed = len(np.intersect1d(correctly_labeled, removed))

        # Mislabeled and kept
        mislabeled = torch.where(weak_labels != gold_labels)[0]
        falsely_kept = len(np.intersect1d(mislabeled, kept))

        # Mislabeled and removed
        correctly_removed = len(np.intersect1d(mislabeled, removed))

        correctly_corrected = 0  # since no other class
        falsely_corrected = 0

    else:
        correctly_labeled = torch.where(weak_labels == gold_labels)[0]
        mislabeled = torch.where(weak_labels != gold_labels)[0]
        assert len(correctly_labeled) + len(mislabeled) == weak_labels.shape[0]

        corrected = torch.where((weak_labels != other) & (chosen_labels == other))[0]
        removed = torch.where(chosen_labels == ignore_index)[0]
        kept = np.where(weak_labels == chosen_labels)[0]
        assert len(corrected) + len(removed) + len(kept) == weak_labels.shape[0]

        gold_other_class = torch.where(gold_labels == other)[0]
        false_other_class = torch.where(gold_labels != other)[0]

        correctly_kept = len(np.intersect1d(correctly_labeled, kept))
        falsely_removed = len(np.intersect1d(correctly_labeled, removed))
        falsely_kept = len(np.intersect1d(mislabeled, kept))
        correctly_removed = len(np.intersect1d(mislabeled, removed))

        correctly_corrected = len(np.intersect1d(corrected, gold_other_class))
        falsely_corrected = len(np.intersect1d(corrected, false_other_class))

    assert correctly_kept + falsely_removed + falsely_kept + correctly_removed + correctly_corrected + falsely_corrected == len(
        weak_labels)

    return correctly_removed, falsely_removed, falsely_kept, correctly_kept, correctly_corrected, falsely_corrected
