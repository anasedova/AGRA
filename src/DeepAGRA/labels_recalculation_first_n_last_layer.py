import logging
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from _scr.autograd_hacks import clear_backprops
from wrench.utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

print_step = 100


def calculate_label(
        batch,
        weak_train_labels,
        comp_batch,
        weak_comp_labels,
        model: Module,
        comparison_criterion: Callable[[Tensor, Tensor], float],
        ignore_index: int = -100,
        other_class: int = None,
        threshold: int = 0,
        device: str = 'cpu',
        layer_aggregation=False,
        batch_num: int = None
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

    # params = [param for param in model.parameters()][-2:]  # weights + biases
    params_first = [param for param in model.parameters()][18:20]
    params_last = [param for param in model.parameters()][194: 196]

    if layer_aggregation is True:
        # flatten comparison gradient layer-wise
        grads_ds_orig_flat_first = [param.grad.clone().detach().flatten().to('cpu') for param in params_first]
        grads_ds_orig_flat_last = [param.grad.clone().detach().flatten().to('cpu') for param in params_last]
    else:
        # fully flatten comparison gradient
        grads_ds_orig_flat_first = torch.cat([param.grad.clone().detach().flatten().to('cpu') for param in params_first])
        grads_ds_orig_flat_last = torch.cat([param.grad.clone().detach().flatten().to('cpu') for param in params_last])

    del params_first
    del params_last

    # create a matrix where future gradients will be stored
    num_ds_instances = weak_train_labels.shape[0]
    # create all labels we want to try out for the batch
    labels_matching = create_label_matching(weak_train_labels.to('cpu'), ignore_index, other_class)
    # create the empty scores matching matrix where the gradients will be stored
    # scores_matching = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    scores_matching_first = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)
    scores_matching_last = np.full((num_ds_instances, labels_matching.shape[1]), threshold, dtype=float)

    label_id = 1
    for hypothetical_labels in labels_matching[:, 1:].T:

        hypothetical_labels = torch.Tensor(hypothetical_labels).long().to(device)

        # compute individual losses and gradients https://github.com/ppmlguy/fastgradclip/blob/0d8bff42ab13fa3471c520a2823050ccf0ff4a21/fastgc/train.py#L46
        model.zero_grad()
        preds = model(batch)
        losses = cross_entropy_with_probs(preds, hypothetical_labels, reduction="none")
        # params = [param for param in model.parameters()][-2:]  # parameters of the last two layers for which we want to compute the gradients
        params_first = [param for param in model.parameters()][18:20]     # parameters of the first layer
        params_last = [param for param in model.parameters()][194:196]    # parameters of the first layer

        # sample_grads = [torch.autograd.grad(losses[i], params, retain_graph=True) for i in range(num_ds_instances)] # takes too much memory

        for sample_id in range(num_ds_instances):
            # don't recompute if same hypothetical label as in last iteration
            if hypothetical_labels[sample_id] == weak_train_labels[sample_id] and label_id > 1:  # other class label already the weak label
                scores_matching_first[sample_id, label_id] = scores_matching_first[sample_id, label_id - 1]
                scores_matching_last[sample_id, label_id] = scores_matching_last[sample_id, label_id - 1]

            else:
                sample_grad_first = torch.autograd.grad(losses[sample_id], params_first, retain_graph=True)
                sample_grad_last = torch.autograd.grad(losses[sample_id], params_last, retain_graph=True)

                if layer_aggregation:
                    # flatten the sample gradient parameter-wise
                    sample_grad_first = [grads.clone().detach().flatten().to('cpu') for grads in sample_grad_first]
                    sample_grad_last = [grads.clone().detach().flatten().to('cpu') for grads in sample_grad_last]

                    # compute average layer-wise similarity score
                    score_first, score_last = 0, 0
                    for i in range(len(sample_grad_first)):
                        score_first += torch.sum(sample_grad_first[i] * grads_ds_orig_flat_first[i]) / (
                                    (torch.norm(sample_grad_first[i]) * torch.norm(grads_ds_orig_flat_first[i])) + 0.000001)
                        score_last += torch.sum(sample_grad_last[i] * grads_ds_orig_flat_last[i]) / (
                                (torch.norm(sample_grad_last[i]) * torch.norm(grads_ds_orig_flat_last[i])) + 0.000001)
                    scores_matching_first[sample_id, label_id] = score_first / len(sample_grad_first)
                    scores_matching_last[sample_id, label_id] = score_last / len(sample_grad_last)
                else:
                    # fully flatten sample-wise gradient
                    sample_grad_first = torch.cat(
                        [grads.clone().detach().flatten().to('cpu') for grads in sample_grad_first])
                    sample_grad_last = torch.cat(
                        [grads.clone().detach().flatten().to('cpu') for grads in sample_grad_last])

                    # compute similarity score
                    scores_matching_first[sample_id, label_id] = torch.sum(sample_grad_first * grads_ds_orig_flat_first) / (
                            (torch.norm(sample_grad_first) * torch.norm(grads_ds_orig_flat_first)) + 0.000001)

                    scores_matching_last[sample_id, label_id] = torch.sum(sample_grad_last * grads_ds_orig_flat_last) / (
                            (torch.norm(sample_grad_last) * torch.norm(grads_ds_orig_flat_last)) + 0.000001)

        if batch_num % print_step == 0:
            print(f"Batch {batch_num}")
            print(f"first_layer_scores: \n {scores_matching_first}")
            print(f"last_layer_scores: \n {scores_matching_last}")

        clear_backprops(model)  # don't think this is necessary
        label_id += 1

    # retrieve label that gave the best similarity score
    max_indices = np.argmax(scores_matching_first, axis=1)
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

# grads = compute_aggregated_grad(preds, hypothetical_labels, params)
# mean_sample_grad = [sum(g) / len(sample_grads) for g in zip(*sample_grads)]
