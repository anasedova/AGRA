import copy
import math
from typing import Callable, Tuple
import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from src.single_label import autograd_hacks
from src.single_label.autograd_hacks import clear_backprops
from wrench.utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

def calculate_label(
        batch,
        weak_train_labels,
        comp_batch,
        weak_comp_labels,
        model: Module,
        comparison_criterion: Callable[[Tensor, Tensor], float],
        ignore_index: int = -100,
        other_class: int = None,
        all_labels: bool = False,
        threshold: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ignore_index: the index that will be assigned
        other_class: the index of the negative class
        all_labels: bool; if set to True, then all classes one after another will be tried in gradient matching
    """

    # calculate the aggregated gradient for the comparison batch
    model.zero_grad() # check difference model.zero_grad(), optimizer.zero_grad(), why optimizer_comp?
    outputs_ds_orig = model(comp_batch)

    loss_ds_orig = comparison_criterion(outputs_ds_orig, weak_comp_labels)
    loss_ds_orig.backward()

    grads_ds_orig_flat = np.concatenate([param.grad.numpy().copy().flatten() for param in model.parameters() if param.requires_grad is True and len(param) > 0]) # gradients wrt to last layer

    # only last layer (if MLP used)
    # last_params = [param for param in model.parameters()][-2:]  # params of final layer
    # grads_ds_orig = [param.grad.numpy().copy() for param in last_params]  # gradients wrt to last layer
    # grads_ds_orig_flat = np.concatenate([grads_ds_orig[0], grads_ds_orig[1].reshape(grads_ds_orig[1].shape[0], 1)],
                                        #axis=1).flatten()

    # compute the gradients for the update batch
    # create a matrix where future gradients will be stored
    num_ds_instances = weak_train_labels.shape[0]

    # create all labels we want to try out for the batch
    labels_matching = create_label_matching(weak_train_labels, ignore_index, all_labels, other_class)

    # create the empty scores matching matrix where the gradients will be stored
    scores_matching = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    autograd_hacks.add_hooks(model)
    output_ds = model(batch)

    # calculate gradients for each sample with different "gold" labels (= neg or ws)
    label_id = 1
    for hypothetical_labels in labels_matching[:, 1:].T: # first column is ignore

        hypothetical_labels = torch.Tensor(hypothetical_labels).long()

        # reset model gradients
        model.zero_grad()
        loss_ds = cross_entropy_with_probs(output_ds, hypothetical_labels, reduction='mean')
        loss_ds.backward(retain_graph=True)

        # compute per-sample gradients
        autograd_hacks.compute_grad1(model)

        # make sure individual losses sum/average up correctly
        # check_grad_reduction(model)

        grads_ds_all = [param.grad1.numpy().copy() for param in model.parameters() if param.requires_grad is True and len(param) > 0]

        # only last two parameters
        # grads_ds_all = [param.grad1.numpy().copy() for param in last_params]
        # maybe make this more efficient
        for sample_id in range(num_ds_instances):

            # retrieve a sample-specific gradient
            sample_grads = [grad[sample_id, ...] for grad in grads_ds_all]
            sample_grads_flat = np.concatenate([grad.flatten() for grad in sample_grads])

            # concatenate weights and bias grad; again for case with only last layer
            # sample_grads_flat = np.concatenate(
                # [sample_grads[0], sample_grads[1].reshape(sample_grads[1].shape[0], 1)], axis=1
            # ).flatten()

            # compute similarity score, maybe use built in scipy?
            scores_matching[sample_id, label_id] = np.sum(sample_grads_flat*grads_ds_orig_flat)/((np.linalg.norm(sample_grads_flat) *
                                                                                        np.linalg.norm(grads_ds_orig_flat)) +
                                                                                        0.000001)

        clear_backprops(model)
        label_id += 1

    max_indices = np.argmax(scores_matching, axis=1)

    # retrieve label that gave the best similarity score
    chosen_labels = torch.tensor(np.take_along_axis(labels_matching, np.expand_dims(max_indices, axis=-1), axis=-1).squeeze(axis=-1))

    # compute grads here and just return them for efficiency, how to do this best with F1 loss?

    return chosen_labels


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

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, predictions, labels):

        if self.num_classes == 2:

            softmax = torch.nn.Softmax(dim=1)

            all_preds = softmax(predictions)
            preds = all_preds[:,1] # probabilities of class 1
            tp = torch.sum(preds * labels)
            fp = torch.sum(preds*(1-labels))
            fn = torch.sum((1-preds)*labels)

            L_F1 = 1-(2 * tp)/(2 * tp + fn + fp + 0.00001)

            if math.isnan(L_F1):
                raise RuntimeError("Loss returns NaN")

        elif self.num_classes > 2:

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

        else:
            raise ValueError("Invalid number of classes")

        return L_F1

def check_grad_reduction(model):

    # make check for last two model parameters
    last_params = [param for param in model.parameters()][-2:]
    for param in last_params:

        if torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06) is False:
            logger.info(f"{param.grad1.mean(dim=0)}, {param.grad}")

        assert (torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06))

