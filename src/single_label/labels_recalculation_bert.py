from typing import Callable, Tuple
import logging

import numpy as np
import torch
import math
import copy
from torch import Tensor
from torch.nn import Module
import collections

from src.single_label.autograd_hacks import clear_backprops
from wrench.utils import cross_entropy_with_probs # todo: change to torch.nn or give as argument

logger = logging.getLogger(__name__)
# todo: this code only works with CE update loss
def calculate_label(
        batch,
        weak_train_labels,
        comp_batch,
        weak_comp_labels,
        model: Module,
        update_criterion: Callable[[Tensor, Tensor], float],
        comparison_criterion: Callable[[Tensor, Tensor], float],
        ignore_index: int = -100,
        other_class: int = None,
        threshold: int = 0,
        device: str = 'cpu'
) -> Tensor:
    """
    Args:
        ignore_index: the index that will be assigned
        other_class: the index of the negative class
    """

    # calculate the aggregated gradient for the comparison batch
    model.zero_grad()
    outputs_ds_orig = model(comp_batch)
    loss_ds_orig = comparison_criterion(outputs_ds_orig, weak_comp_labels)
    loss_ds_orig.backward()

    params = [param for param in model.parameters()][-2]  # weights of last layer

    grads_ds_orig_flat = params.grad.clone().detach().to('cpu').flatten()

    del params

    # create a matrix where future gradients will be stored
    num_ds_instances = weak_train_labels.shape[0]
    # create all labels we want to try out for the batch
    labels_matching = create_label_matching(weak_train_labels.to('cpu'), ignore_index, other_class)
    # create the empty scores matching matrix where the gradients will be stored
    scores_matching = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    label_id = 1
    preds = model(batch)
    for hypothetical_labels in labels_matching[:, 1:].T:

        hypothetical_labels = torch.Tensor(hypothetical_labels).long().to(device)

        model.zero_grad()
        # compute individual losses and gradients https://github.com/ppmlguy/fastgradclip/blob/0d8bff42ab13fa3471c520a2823050ccf0ff4a21/fastgc/train.py#L46
        losses = cross_entropy_with_probs(preds, hypothetical_labels, reduction='none') # coincides with torch.nn.CrossEntropyLoss
        params = [param for param in model.parameters()][-2]

        # sample_grads = [torch.autograd.grad(losses[i], params, retain_graph=True) for i in range(num_ds_instances)] # takes too much memory

        for sample_id in range(num_ds_instances):
            # don't recompute if same hypothetical label as in last iteration
            if hypothetical_labels[sample_id] == weak_train_labels[sample_id] and label_id > 1:  # other class label already the weak label
                scores_matching[sample_id, label_id] = scores_matching[sample_id, label_id - 1]

            else:
                sample_grad = torch.autograd.grad(losses[sample_id], params, retain_graph=True)[0]

                # fully flatten sample-wise gradient
                sample_grad = sample_grad.clone().detach().to('cpu').flatten()

                # compute similarity score
                scores_matching[sample_id, label_id] = torch.sum(sample_grad*grads_ds_orig_flat)/((torch.norm(sample_grad)*torch.norm(grads_ds_orig_flat)) + 0.000001)

        clear_backprops(model) # don't think this is necessary
        label_id += 1

    del params

    # retrieve label that gave the best similarity score
    max_indices = np.argmax(scores_matching, axis=1)
    chosen_labels = torch.tensor(
        np.take_along_axis(labels_matching, np.expand_dims(max_indices, axis=-1), axis=-1).squeeze(axis=-1))

    return chosen_labels

def create_label_matching(
        labels_ds_with_sub, ignore_index: int, other_class: int = None
) -> np.ndarray:
    num_ds_instances = labels_ds_with_sub.shape[0]
    if other_class is not None:
        labels_matching = np.array([[ignore_index, other_class, other_class], ] * num_ds_instances, dtype=np.int64)
        labels_matching[:, 1] = labels_ds_with_sub
        return labels_matching
    else:
        labels_matching = np.array([[ignore_index, ignore_index], ] * num_ds_instances, dtype=np.int64)
        labels_matching[:, 1] = labels_ds_with_sub
        return labels_matching

def compute_aggregated_grad(preds, labels, params):
    loss = cross_entropy_with_probs(preds, labels, reduction="mean")
    loss.backward()
    grads = [param.grad for param in params if param.requires_grad and len(param) > 0]

    return grads

class F1Loss():

    def __init__(self, num_classes, avg="macro"):
        self.num_classes = num_classes
        self.avg = avg

    def __call__(self, predictions, labels):

        if self.avg == "binary":

            softmax = torch.nn.Softmax(dim=1)

            all_preds = softmax(predictions)
            preds = all_preds[:, 1] # probabilities of class 1
            tp = torch.sum(preds * labels)
            fp = torch.sum(preds*(1-labels))
            fn = torch.sum((1-preds)*labels)

            L_F1 = 1-(2 * tp)/(2 * tp + fn + fp + 0.00001)

            if math.isnan(L_F1):
                raise RuntimeError("Loss returns NaN")

        if self.avg == "macro":

            # define soft-max
            softmax = torch.nn.Softmax(dim=1)
            f1 = torch.zeros(self.num_classes)  # storing location for f1 of each class
            preds = softmax(predictions)  # probability for each class for each example

            for label in range(0, self.num_classes):

                # binarize labels -> 1: belongs to label, 0: does not belong to label
                labels_bin = copy.deepcopy(labels)
                labels_bin = torch.where(labels_bin == label, 1, 0)

                # compute the f1 score for the class
                tp = torch.sum(preds[:, label] * labels_bin)
                fp = torch.sum(preds[:, label] * (1 - labels_bin))
                fn = torch.sum((1 - preds[:, label]) * labels_bin)

                f1[label] = (2 * tp)/(2 * tp + fn + fp + 0.00001)

            # define loss 1-macro_F1
            L_F1 = 1 - torch.mean(f1)

            if math.isnan(L_F1) or math.isinf(L_F1):
                raise RuntimeError("Loss returns NaN or Inf")

        return L_F1