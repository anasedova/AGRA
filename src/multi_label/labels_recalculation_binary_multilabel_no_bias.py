from typing import Callable, Union
import copy

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.multi_label import autograd_hacks


def calculate_label(
        inputs_ds_orig: Tensor,
        labels_ds_orig: Tensor,
        model: Module,
        optimizer: Optimizer,
        comparison_criterion: Callable[[Tensor, Tensor], float],
        inputs_ds_with_sub: Tensor,
        labels_ds_with_sub: Tensor,
        num_classes: int,
        ignore_index: int = -100,
        loss_type: str = "mean",
        threshold: int = 0
) -> Union[Tensor, np.ndarray]:

    """
    this version computes gradients for all samples and sets those corresponding to ignored labels to zero
    """

    # thresh = scipy.stats.norm.ppf(threshold)

    # calculate the gradient with the original weak labels
    optimizer.zero_grad()
    outputs_ds_orig = torch.sigmoid(model(inputs_ds_orig))

    loss_ds_orig = comparison_criterion(outputs_ds_orig, labels_ds_orig) # criterion
    loss_ds_orig.backward()
    grads_ds_orig = [param.grad.numpy().copy() for param in model.parameters()]

    # f1_grad_check_macro(outputs_ds_orig, labels_ds_orig, inputs_ds_orig, grads_ds_orig)

    # concatenate weights and bias gradients
    grads_ds_orig = grads_ds_orig[0]

    num_ds_instances = inputs_ds_with_sub.shape[0]

    # labels chosen for training; some will be substituted in the following
    chosen_labels = copy.deepcopy(labels_ds_with_sub)

    # compute the gradient of each individual sample
    autograd_hacks.add_hooks(model)

    model.zero_grad()
    output_ds = torch.sigmoid(model(inputs_ds_with_sub))
    loss_ds = comparison_criterion(output_ds, labels_ds_with_sub)
    loss_ds.backward(retain_graph=True)
    autograd_hacks.compute_grad1(model, loss_type=loss_type)

    # check if aggregated gradient close to sum/mean of individual gradients
    for param in model.parameters():

        if loss_type == "mean":

            if torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06) is False:
                print(param.grad1.mean(dim=0), param.grad)

            assert (torch.allclose(param.grad1.mean(dim=0), param.grad, atol=1e-06))

        if loss_type == "sum":

            if torch.allclose(param.grad1.sum(dim=0), param.grad, atol=1e-06) is False:
                print(param.grad1.sum(dim=0), param.grad)

            assert (torch.allclose(param.grad1.sum(dim=0), param.grad, atol=1e-06))

    grads_ds_all = [params.grad1.numpy().copy() for params in model.parameters()]

    # compute inner product of gradients corresponding to the same output node
    # scores_matching = np.full((num_ds_instances, labels_ds_orig.shape[1]), 0, dtype=float)
    # compute inner product of gradients corresponding to the same output node
    for sample_id in range(num_ds_instances):
        sample_grads = [grad[sample_id, ...] for grad in grads_ds_all][0]  # [grad[sample_id, ...] for grad in grads_ds_all] # grads_ds_all[sample_id] # len 2: num_classes x num_features and num_classes

        # compute similarity of each gradient to the gradient of the comparison batch
        scores = np.sum(sample_grads * grads_ds_orig, axis=1) / (
                (np.linalg.norm(sample_grads, axis=1) * np.linalg.norm(grads_ds_orig, axis=1)) + 0.000001)
        (chosen_labels[sample_id, :])[scores <= threshold] = ignore_index

    return chosen_labels


class F1LossMulti:

    def __init__(self, ignore_index=None, reduction="macro"):
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, predictions, labels):

        exp_tp = predictions * labels
        exp_fp = predictions * (1 - labels)
        exp_fn = (1 - predictions) * labels

        if self.ignore_index is not None:  # set values of ignored samples to 0
            mask = torch.where(labels == self.ignore_index)

            exp_tp[mask] = 0
            exp_fp[mask] = 0
            exp_fn[mask] = 0

        if self.reduction == "macro":

            tp = torch.sum(exp_tp, dim=0)  # tp per class # predictions*labels -> shape [32, 14]
            fp = torch.sum(exp_fp, dim=0)
            fn = torch.sum(exp_fn, dim=0)

            f1_per_class = (2 * tp) / (2 * tp + fn + fp + 0.00001) # stabilizer

            # if nan -> tp, fn, fp all 0 -> denominator 0, want to not make update for that class
            # f1_per_class = torch.nan_to_num(f1_per_class, nan=1) # set class F1 to 1

            L_F1 = torch.mean(1-f1_per_class) # mean

        if self.reduction == "micro":

            # initialize tp, fp, fn
            tp = 0
            fp = 0
            fn = 0

            # compute the values aggregated over all examples and classes
            tp += torch.sum(exp_tp) # aggregated sum over all dimensions (labels, samples)
            fp += torch.sum(exp_fp)
            fn += torch.sum(exp_fn)

            L_F1 = 1 - (2 * tp) / (2 * tp + fn + fp + 0.00001)  # 1 - micro F1, + 0.00001

        return L_F1


class MaskedBCELoss:

    def __init__(self, ignore_index, num_classes, reduction="avg_per_class"):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.reduction = reduction

    def __call__(self, predictions, labels):

        # compute unreduced loss per sample
        loss = nn.BCELoss(reduction='none')
        losses = loss(predictions, labels)  # batch size x num classes

        # mask the contributions of entries that correspond to elements that were assigned ignore_index
        losses[torch.where(labels == self.ignore_index)] = 0

        # reduce the loss
        if self.reduction == "sum":  # not really sensible
            return torch.sum(losses)

        if self.reduction == "avg_per_class":

            # class_avg version
            retained = torch.where(labels != torch.tensor(self.ignore_index))
            retained_per_class = torch.bincount(retained[1], minlength=self.num_classes)

            retained_per_class[retained_per_class == 0] = 1

            return torch.sum(losses/retained_per_class)/self.num_classes

        if self.reduction == "avg_per_sample":
            # sample_avg version
            retained = torch.where(labels != torch.tensor(self.ignore_index))
            retained_per_sample = torch.bincount(retained[0], minlength=len(losses))
            retained_per_sample[retained_per_sample == 0] = 1
            return torch.sum(losses.T / retained_per_sample)

        if self.reduction == "avg_all": # not really sensible
            # alt version
            n_non_zero = torch.count_nonzero(losses)
            return (1 / n_non_zero) * torch.sum(losses)


