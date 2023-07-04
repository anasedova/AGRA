from typing import Callable, Union
import copy
import logging

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module

from src import autograd_hacks

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_label_multilabel(
        batch,
        weak_train_labels,
        comp_batch,
        weak_comp_labels,
        model: Module,
        comparison_criterion: Callable[[Tensor, Tensor], float],
        ignore_index: int = -100,
        loss_type: str = "mean",
        threshold: float = 0.0,
        include_bias: bool = False
) -> Union[Tensor, np.ndarray]:

    # compute gradient of comparison batch
    weak_comp_labels = weak_comp_labels.float().to(device)
    model.zero_grad()
    outputs_ds_orig = torch.sigmoid(model(comp_batch["features"].to(device)))
    loss_ds_orig = comparison_criterion(outputs_ds_orig, weak_comp_labels) # criterion
    loss_ds_orig.backward()
    grads_ds_orig = [param.grad.detach().clone().cpu().numpy() for param in model.parameters()]

    # concatenate weights and bias gradients
    if include_bias is False:
        grads_ds_orig = grads_ds_orig[0] # #labels x #embedding dimension
    else:
        grads_ds_orig = np.concatenate([grads_ds_orig[0], grads_ds_orig[1].reshape(grads_ds_orig[1].shape[0], 1)], axis=1)

    # labels chosen for training; some will be substituted in the following
    chosen_labels = copy.deepcopy(weak_train_labels)

    # compute the gradient of each individual sample
    autograd_hacks.add_hooks(model)

    model.zero_grad()
    output_ds = torch.sigmoid(model(batch["features"].to(device)))
    loss_ds = comparison_criterion(output_ds, weak_train_labels.float().to(device))
    loss_ds.backward(retain_graph=True)
    autograd_hacks.compute_grad1(model, loss_type=loss_type)

    grads_ds_all = [params.grad1.detach().clone().cpu().numpy() for params in model.parameters()]

    num_ds_instances = weak_train_labels.shape[0]
    for sample_id in range(num_ds_instances):
        sample_grads = [grad[sample_id, ...] for grad in grads_ds_all]

        if include_bias is False:
            sample_grads = sample_grads[0]
        else:
            sample_grads = np.concatenate([sample_grads[0], sample_grads[1].reshape(sample_grads[1].shape[0], 1)], axis=1)

        # compute similarity of each gradient to the gradient of the comparison batch
        scores = np.sum(sample_grads * grads_ds_orig, axis=1) / ((np.linalg.norm(sample_grads, axis=1) * np.linalg.norm(grads_ds_orig, axis=1)) + 0.000001)
        (chosen_labels[sample_id, :])[scores <= threshold] = ignore_index

    return chosen_labels


class MaskedBCELoss:

    def __init__(self, ignore_index, num_classes):
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, predictions, labels):

        # compute unreduced loss per sample
        loss = nn.BCELoss(reduction='none')
        losses = loss(predictions, labels)  # batch size x num classes

        # mask the contributions of entries that correspond to elements that were assigned ignore_index
        losses[torch.where(labels == self.ignore_index)] = 0

        # class_avg version
        retained = torch.where(labels != torch.tensor(self.ignore_index))
        retained_per_class = torch.bincount(retained[1], minlength=self.num_classes)

        retained_per_class[retained_per_class == 0] = 1 # avoid division by zero

        return torch.sum(losses/retained_per_class)/self.num_classes