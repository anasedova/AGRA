from typing import Callable, Tuple
import logging

import numpy as np
import torch
import math
import copy
from torch import Tensor
from torch.nn import Module

from src.single_label import autograd_hacks
from src.single_label.autograd_hacks import clear_backprops

logger = logging.getLogger(__name__)


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
        device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Args:
        ignore_index: the index that will be assigned
        other_class: the index of the negative class
        all_labels: bool; if set to True, then all classes one after another will be tried in gradient matching
    """

    # calculate the gradient for the comparison batch
    model.zero_grad()
    outputs_ds_orig = model(comp_batch)
    loss_ds_orig = comparison_criterion(outputs_ds_orig, weak_comp_labels)
    loss_ds_orig.backward()
    grads_ds_orig = [param.grad.detach().clone().cpu().numpy().flatten() for param in model.parameters() if param.requires_grad and len(param) > 0]

    # only take gradient with respect to weight matrix
    grads_ds_orig = grads_ds_orig[0]

    # create a matrix where future gradients will be stored
    num_ds_instances = weak_train_labels.shape[0]

    # create all labels we want to try out for the batch
    labels_matching = create_label_matching(weak_train_labels.to('cpu'), ignore_index, other_class)

    # create the empty scores matching matrix where the gradients will be stored
    scores_matching = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    autograd_hacks.add_hooks(model)
    output_ds = model(batch)

    # calculate gradients for each sample with different "gold" labels (= neg or ws)
    label_id = 1
    for hypothetical_labels in labels_matching[:, 1:].T: # first column is ignore

        hypothetical_labels = torch.Tensor(hypothetical_labels).long().to(device)

        # reset model gradients
        model.zero_grad()
        loss_ds = comparison_criterion(output_ds, hypothetical_labels)
        loss_ds.backward(retain_graph=True)

        # compute per-sample gradients
        autograd_hacks.compute_grad1(model)

        grads_ds_all = [params.grad1.detach().clone().cpu().numpy() for params in model.parameters() if params.requires_grad and len(params) > 0]

        for sample_id in range(num_ds_instances):

            # retrieve a sample-specific gradient
            sample_grads = [grad[sample_id, ...] for grad in grads_ds_all]

            # concatenate weights and bias grad
            sample_grads = sample_grads[0].flatten() # only take gradient with respect to weight matrix

            # compute similarity score
            scores_matching[sample_id, label_id] = np.sum(sample_grads*grads_ds_orig)/((np.linalg.norm(sample_grads) *
                                                                                        np.linalg.norm(grads_ds_orig)) +
                                                                                        0.000001)

        clear_backprops(model)
        label_id += 1

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