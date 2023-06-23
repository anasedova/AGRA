import copy
import math

import torch


class F1Loss:
    """class that implements the proposed versions of the F1 loss for single-label datasets"""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, predictions, labels):

        if self.num_classes == 2:

            softmax = torch.nn.Softmax(dim=1)
            all_preds = softmax(predictions)
            preds = all_preds[:, 1]
            tp = torch.sum(preds * labels)
            fp = torch.sum(preds*(1-labels))
            fn = torch.sum((1-preds)*labels)

            # tp + fn = number positive labels
            L_F1 = 1-(2 * tp)/(2 * tp + fn + fp + 0.00001)

            if math.isnan(L_F1):
                raise RuntimeError("Loss returns NaN")

        elif self.num_classes > 2:

            # define soft-max
            softmax = torch.nn.Softmax(dim=1)
            f1 = torch.zeros(self.num_classes)
            preds = softmax(predictions)

            for label in range(0, self.num_classes):

                labels_bin = copy.deepcopy(labels)
                labels_bin = torch.where(labels_bin == label, 1, 0)

                # compute the f1 score for the class
                tp = torch.sum(preds[:, label] * labels_bin)
                fp = torch.sum(preds[:, label] * (1 - labels_bin))
                fn = torch.sum((1 - preds[:, label]) * labels_bin)

                f1[label] = (2 * tp)/(2 * tp + fn + fp + 0.00001)

            # define loss as 1-macro_F1
            L_F1 = 1 - torch.mean(f1)

            if math.isnan(L_F1) or math.isinf(L_F1):
                raise RuntimeError("Loss returns NaN or Inf")

        else:
            raise ValueError("Invalid number of classes")

        return L_F1

class F1LossMulti:
    """class that implements the macro-F1 loss for multi-label datasets"""

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, predictions, labels):

        exp_tp = predictions * labels
        exp_fp = predictions * (1 - labels)
        exp_fn = (1 - predictions) * labels

        if self.ignore_index is not None:  # set values of ignored samples to 0
            mask = torch.where(labels == self.ignore_index)

            exp_tp[mask] = 0
            exp_fp[mask] = 0
            exp_fn[mask] = 0

        tp = torch.sum(exp_tp, dim=0)
        fp = torch.sum(exp_fp, dim=0)
        fn = torch.sum(exp_fn, dim=0)

        f1_per_class = (2 * tp) / (2 * tp + fn + fp + 0.00001)

        L_F1 = torch.mean(1-f1_per_class)

        if math.isnan(L_F1) or math.isinf(L_F1):
            raise RuntimeError("Loss returns NaN or Inf")

        return L_F1
