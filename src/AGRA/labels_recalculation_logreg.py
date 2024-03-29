import logging
from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from src import autograd_hacks
from src.autograd_hacks import clear_backprops

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def calculate_label(
        batch,
        weak_train_labels,
        comp_batch,
        weak_comp_labels,
        model: Module,
        comparison_criterion: Callable[[Tensor, Tensor], float],
        ignore_index: int = -100,
        other_class: int = None,
        threshold: float = 0,
        include_bias: bool = True,
        adaptive_threshold: bool = False,
        loss_type: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ignore_index: the index that will be assigned
        other_class: the index of the negative class
        all_labels: bool; if set to True, then all classes one after another will be tried in gradient matching
    """

    # comp_batch, weak_comp_labels, model = comp_batch.to(device), weak_comp_labels.to(device), model.to(device)
    # calculate the gradient for the comparison batch
    weak_comp_labels = weak_comp_labels.to(device)
    model.zero_grad()
    outputs_ds_orig = model(comp_batch["features"].to(device))
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
    output_ds = model(batch["features"].to(device))

    # calculate gradients for each sample with different "gold" labels (= neg or ws)
    label_id = 1
    for hypothetical_labels in labels_matching[:, 1:].T: # first column is ignore

        hypothetical_labels = torch.Tensor(hypothetical_labels).long().to(device)

        # reset model gradients
        model.zero_grad()
        loss_ds = comparison_criterion(output_ds, hypothetical_labels)
        loss_ds.backward(retain_graph=True)

        # compute per-sample gradients
        autograd_hacks.compute_grad1(model, loss_type=loss_type)

        grads_ds_all = [params.grad1.detach().clone().cpu().numpy() for params in model.parameters() if
                        params.requires_grad and len(params) > 0]

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
