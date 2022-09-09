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

    # calculate the gradient with the original weak labels
    optimizer.zero_grad()
    outputs_ds_orig = torch.sigmoid(model(inputs_ds_orig))

    loss_ds_orig = comparison_criterion(outputs_ds_orig, labels_ds_orig) # criterion
    loss_ds_orig.backward()
    grads_ds_orig = [param.grad.numpy().copy() for param in model.parameters()]

    # f1_grad_check_macro(outputs_ds_orig, labels_ds_orig, inputs_ds_orig, grads_ds_orig)

    # concatenate weights and bias gradients
    grads_ds_orig = np.concatenate([grads_ds_orig[0], grads_ds_orig[1].reshape(grads_ds_orig[1].shape[0], 1)], axis=1)

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
    for sample_id in range(num_ds_instances):
        sample_grads = [grad[sample_id, ...] for grad in grads_ds_all] # len 2: num_classes x num_features and num_classes

        # concatenate weights and bias grad
        sample_grads = np.concatenate([sample_grads[0], sample_grads[1].reshape(sample_grads[1].shape[0], 1)], axis=1)

        # compute similarity of each gradient to the gradient of the comparison batch
        for class_id in range(num_classes):
            score = np.sum(sample_grads[class_id]*grads_ds_orig[class_id])/((np.linalg.norm(sample_grads[class_id]) * np.linalg.norm(grads_ds_orig[class_id])) + 0.000001)

            # if the similarity score is too small -> ignore label
            if score <= threshold: # threshold
                chosen_labels[sample_id, class_id] = ignore_index

    return chosen_labels


def masked_BCE_loss(predictions, targets, ignore_index, reduction="avg_per_class", num_classes=13):

    # compute unreduced loss
    loss = nn.BCELoss(reduction='none')
    losses = loss(predictions, targets)  # batch size x num classes

    # mask the contributions of entries that correspond to elements that were assigned ignore_index
    losses[torch.where(targets == ignore_index)] = 0

    # reduce the loss
    if reduction == "sum":  # not used
        return torch.sum(losses)

    if reduction == "avg_per_class":
        retained = torch.where(targets != torch.tensor(ignore_index))
        retained_per_class = torch.bincount(retained[1], minlength=num_classes)
        retained_per_class[retained_per_class == 0] = 1
        return torch.sum(losses/retained_per_class) # /len(targets) # without len targets?

    if reduction == "avg_per_sample":
        retained = torch.where(targets != torch.tensor(ignore_index))
        retained_per_sample = torch.bincount(retained[0], minlength=len(losses))
        retained_per_sample[retained_per_sample == 0] = 1
        return torch.sum(losses.T/retained_per_sample)

    if reduction == "avg_all":
        n_non_zero = torch.count_nonzero(losses)
        return (1/n_non_zero)*torch.sum(losses)


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

            # handle classes for which all labels were ignored
            # zero_classes = torch.where(retained_per_class==0)[0]
            # if(len(zero_classes > 0)):
                # self.num_classes = self.num_classes - len(zero_classes > 0)

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


def f1_grad_check_macro(predictions, labels, inputs, grads_og):

    exp_tp = predictions * labels
    exp_fp = predictions * (1 - labels)
    exp_fn = (1 - predictions) * labels

    tp = torch.sum(exp_tp, dim=0)  # tp per class # predictions*labels -> shape [128, 12]
    fp = torch.sum(exp_fp, dim=0)
    fn = torch.sum(exp_fn, dim=0)

    grads_w = np.zeros([labels.shape[1], inputs.shape[1]])
    grads_b = np.zeros(labels.shape[1])
    grads_w_sample = np.zeros([inputs.shape[0], labels.shape[1], inputs.shape[1]])
    grads_b_sample = np.zeros([inputs.shape[0], inputs.shape[1]])

    tp_grad = labels*predictions*(1-predictions) # 16 x 12
    fp_grad = (1-labels)*predictions*(1-predictions)
    fn_grad = -labels*predictions*(1-predictions)

    for label_id in range(0, labels.shape[1]):
        tp_c = tp_grad[:,label_id]
        fp_c = fp_grad[:,label_id]
        fn_c = fn_grad[:,label_id]
        for feature_id in range(inputs.shape[1]):
            inputs_i = inputs[:, feature_id]
            # grads_w[label_id, feature_id] = (1/labels.shape[1])*(-2*(torch.sum(tp_c*inputs_i)*(2*tp[label_id]+fp[label_id]+fn[label_id])-(2*torch.sum(tp_c*inputs_i) + torch.sum(fn_c*inputs_i) + torch.sum(fp_c*inputs_i))*tp[label_id])/(pow(2*tp[label_id]+fn[label_id]+fp[label_id], 2)))

            grads_w[label_id, feature_id] = torch.sum((1 / labels.shape[1]) * (-2 * (
                        tp_c * inputs_i * (2 * tp[label_id] + fp[label_id] + fn[label_id]) - (
                            2 * tp_c * inputs_i + fn_c * inputs_i + fp_c * inputs_i) *
                        tp[label_id]) / (pow(2 * tp[label_id] + fn[label_id] + fp[label_id], 2))))

            grads_w_sample[:, label_id, feature_id] = ((1 / labels.shape[1]) * (-2 * (
                        tp_c * inputs_i * (2 * tp[label_id] + fp[label_id] + fn[label_id] + 0.00001) - (
                            2 * tp_c * inputs_i + fn_c * inputs_i + fp_c * inputs_i) *
                        tp[label_id]) / (pow(2 * tp[label_id] + fn[label_id] + fp[label_id] + 0.00001, 2)))).detach().numpy()


def f1_grad_check_BCE(predictions, labels, inputs, grads_og):

    # check whether BCE gradient matches derived formula

    g_w = np.zeros([inputs.shape[0], 12, inputs.shape[1]])
    g_b = np.zeros([inputs.shape[0], 12])

    for sample_id in range(inputs.shape[0]):
        for feature_id in range(inputs.shape[1]):
            for label_id in range(12):
                g_w[sample_id, label_id, feature_id] = predictions[sample_id, label_id]*inputs[sample_id, feature_id]*(1-labels[sample_id, label_id])+(predictions-1)[sample_id, label_id]*inputs[sample_id, feature_id]*labels[sample_id, label_id]
                g_b[sample_id, label_id] = predictions[sample_id, label_id]*(1-labels[sample_id, label_id])+(predictions-1)[sample_id, label_id]*labels[sample_id, label_id]

    g_w = 1/12*g_w
    g_b = 1/12*g_b

    print(np.allclose(g_w, grads_og[0]))
    print(np.allclose(g_b, grads_og[1]))


def f1_grad_check_micro(predictions, labels, inputs, grads_og):

    exp_tp = predictions * labels
    exp_fp = predictions * (1 - labels)
    exp_fn = (1 - predictions) * labels

    # estimated confusion matrix entries disregarding classes
    tp = torch.sum(exp_tp)
    fp = torch.sum(exp_fp)
    fn = torch.sum(exp_fn)

    grads_w = np.zeros([inputs.shape[0], labels.shape[1], inputs.shape[1]]) # 1280 x12
    grads_b = np.zeros(labels.shape[1])

    tp_grad = labels*predictions*(1-predictions)
    fp_grad = (1-labels)*predictions*(1-predictions)
    fn_grad = -labels*predictions*(1-predictions)

    for label_id in range(0, labels.shape[1]):
        tp_c = tp_grad[:, label_id]
        fp_c = fp_grad[:, label_id]
        fn_c = fn_grad[:, label_id]
        for feature_id in range(0, inputs.shape[1]):
            inputs_i = inputs[:, feature_id]
            grads_w[:, label_id, feature_id] = ((-2 * (
                tp_c * inputs_i * (2 * tp + fp + fn + 0.00001) - (
                2 * tp_c * inputs_i + fn_c * inputs_i + fp_c * inputs_i) *
                tp) / (pow(2 * tp + fn + fp + 0.00001, 2)))).detach().numpy()

    print(np.allclose(grads_w, grads_og[0]))


