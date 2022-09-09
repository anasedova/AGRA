import copy
import math
from typing import Callable, Tuple
import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.single_label import autograd_hacks
from src.single_label.autograd_hacks import clear_backprops

logger = logging.getLogger(__name__)


def calculate_label(
        inputs_ds_orig: Tensor,
        labels_ds_orig: Tensor,
        model: Module,
        optimizer_comp: Optimizer,
        criterion: Callable[[Tensor, Tensor], float],
        inputs_ds_with_sub: Tensor,
        labels_ds_with_sub: Tensor,
        ignore_index: int = -100,
        other_class: int = None,
        all_labels: bool = False,
        threshold: int = 0,
        loss_type: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ignore_index: the index that will be assigned
        other_class: the index of the negative class
        all_labels: bool; if set to True, then all classes one after another will be tried in gradient matching
    """

    # calculate the gradient for the comparison batch
    optimizer_comp.zero_grad()
    outputs_ds_orig = model(inputs_ds_orig)

    loss_ds_orig = criterion(outputs_ds_orig, labels_ds_orig)
    loss_ds_orig.backward()
    grads_ds_orig = [param.grad.numpy().copy() for param in model.parameters()]

    # concatenate weights and bias gradients
    grads_ds_orig = np.concatenate([grads_ds_orig[0], grads_ds_orig[1].reshape(grads_ds_orig[1].shape[0], 1)], axis=1)

    # create a matrix where future gradients will be stored
    num_ds_instances = inputs_ds_with_sub.shape[0]

    # create all labels we want to try out for the batch
    labels_matching = create_label_matching(labels_ds_with_sub, ignore_index, all_labels, other_class)

    # create the empty scores matching matrix where the gradients will be stored
    scores_matching = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    autograd_hacks.add_hooks(model)
    output_ds = model(inputs_ds_with_sub)

    # calculate gradients for each sample with different "gold" labels (= neg or ws)
    label_id = 1
    for hypothetical_labels in labels_matching[:, 1:].T: # first column is ignore

        hypothetical_labels = torch.Tensor(hypothetical_labels).long()

        # reset model gradients
        model.zero_grad()
        loss_ds = criterion(output_ds, hypothetical_labels)
        loss_ds.backward(retain_graph=True)

        # compute per-sample gradients
        autograd_hacks.compute_grad1(model, loss_type=loss_type)

        for param in model.parameters():

            if loss_type == "mean":
                if torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06) is False:
                    logger.info(f"{param.grad1.mean(dim=0)}, {param.grad}")

                assert (torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06))

            if loss_type == "sum":
                if torch.allclose(param.grad1.sum(dim=0), param.grad, atol=1e-06) is False:
                    logger.info(f"{param.grad1.mean(dim=0)}, {param.grad}")

                assert (torch.allclose(param.grad1.sum(dim=0), param.grad, atol=1e-06))

        grads_ds_all = [params.grad1.numpy().copy() for params in model.parameters()]

        for sample_id in range(num_ds_instances):

            # retrieve a sample-specific gradient
            sample_grads = [grad[sample_id, ...] for grad in grads_ds_all]

            # concatenate weights and bias grad
            sample_grads = np.concatenate(
                [sample_grads[0], sample_grads[1].reshape(sample_grads[1].shape[0], 1)], axis=1
            )

            # compute similarity score
            scores_matching[sample_id, label_id] = np.sum(sample_grads*grads_ds_orig)/((np.linalg.norm(sample_grads) *
                                                                                        np.linalg.norm(grads_ds_orig)) +
                                                                                       0.000001)
        clear_backprops(model)
        label_id += 1

    max_indices = np.argmax(scores_matching, axis=1)

    # retrieve label that gave the best similarity score
    chosen_labels = np.take_along_axis(labels_matching, np.expand_dims(max_indices, axis=-1), axis=-1).squeeze(axis=-1)

    return chosen_labels, labels_matching


def create_label_matching(
        labels_ds_with_sub, ignore_index: int, all_labels: bool, other_class: int = None
) -> np.ndarray:
    num_ds_instances = labels_ds_with_sub.shape[0]
    if other_class is not None:
        labels_matching = np.array([[ignore_index, other_class, other_class], ] * num_ds_instances, dtype=np.int64)
        return add_columns_for_classes(labels_ds_with_sub, labels_matching, num_ds_instances, all_labels)
    else:
        labels_matching = np.array([[ignore_index, ignore_index], ] * num_ds_instances, dtype=np.int64)
        return add_columns_for_classes(labels_ds_with_sub, labels_matching, num_ds_instances, all_labels)


def add_columns_for_classes(
        labels_ds_with_sub: Tensor, labels_matching: np.ndarray, num_ds_instances: int, all_labels: bool
) -> np.ndarray:
    if all_labels:
        num_classes = int(max(labels_ds_with_sub) + 1)
        for i in range(num_classes):
            labels_matching = np.column_stack(
                (labels_matching, np.array([[i], ] * num_ds_instances, dtype=np.int64))
            )
        labels_matching[:, 1] = labels_ds_with_sub
    else:
        labels_matching[:, -1] = labels_ds_with_sub
    return labels_matching

class F1Loss():

    def __init__(self, num_classes, avg="macro"):
        self.num_classes = num_classes
        self.avg = avg

    def __call__(self, predictions, labels):

        if self.avg == "binary":

            softmax = torch.nn.Softmax(dim=1)

            all_preds = softmax(predictions)
            preds = all_preds[:,1] # probabilities of class 1
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
