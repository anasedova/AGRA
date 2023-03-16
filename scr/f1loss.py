import torch
import math
import copy


class F1Loss:
    """ class that implements the proposed versions of the F1 loss for single-label datasets """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, predictions, labels):

        if self.num_classes == 2:

            softmax = torch.nn.Softmax(dim=1)
            all_preds = softmax(predictions)
            preds = all_preds[:, 1]     # probabilities of class 1
            tp = torch.sum(preds * labels)      # sum of probs of belonging to pos. assigned to truly positive samples
            fp = torch.sum(preds*(1-labels))  # sum of probs of belonging to pos. assigned to negative samples
            fn = torch.sum((1-preds)*labels)  # sum of probs of belonging to neg. assigned to positive samples

            # tp + fn = number positive labels
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
                tp = torch.sum(preds[:, label] * labels_bin) # sum of probs that elements that belong to this label are predicted this label
                fp = torch.sum(preds[:, label] * (1 - labels_bin)) # sum of probs that elements that do not belong to this labels are predicted this label
                fn = torch.sum((1 - preds[:, label]) * labels_bin) # sum of probs that elements belonging to this label are classified differently

                f1[label] = (2 * tp)/(2 * tp + fn + fp + 0.00001)

            # define loss 1-macro_F1
            L_F1 = 1 - torch.mean(f1)

            if math.isnan(L_F1) or math.isinf(L_F1):
                raise RuntimeError("Loss returns NaN or Inf")

        else:
            raise ValueError("Invalid number of classes")

        return L_F1
