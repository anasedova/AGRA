import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from src.single_label.utils import AGRADataset

logger = logging.getLogger(__name__)


def eval_grad_match(dev_X, dev_y, net, avg="macro", return_preds=False):
    dev_ds = AGRADataset(dev_X, dev_y)

    with torch.no_grad():
        dev_prediction_probas = net.forward(dev_ds.features)
        dev_predictions = torch.argmax(dev_prediction_probas, dim=1)

    metrics = classification_report(y_true=dev_y, y_pred=dev_predictions.detach().numpy(), output_dict=True)

    if avg == "macro":
        if return_preds is False:
            return metrics["accuracy"], metrics['macro avg']['f1-score'], metrics['macro avg']['precision'], \
                   metrics['macro avg']['recall']
        else:
            return metrics["accuracy"], metrics['macro avg']['f1-score'], metrics['macro avg']['precision'], \
                   metrics['macro avg']['recall'], dev_predictions

    elif avg == "binary":
        if return_preds is False:
            return metrics["accuracy"], metrics['1']['f1-score'], metrics['1']['precision'], metrics['1']['recall']
        else:
            return metrics["accuracy"], metrics['1']['f1-score'], metrics['1']['precision'], metrics['1'][
                'recall'], dev_predictions


def collect_statistics(
        num_ds_instances: int, ignore_index: int, chosen_labels: np.ndarray, ds_labels_matching: np.ndarray,
        no_relation_index: int = None
):
    # initialize the counts
    norel_kept_batch, norel_ignored_batch, ds_kept_batch, ds_to_norel_batch, ds_ignored_batch = 0, 0, 0, 0, 0

    for i in range(num_ds_instances):
        chosen_label = int(chosen_labels[i])
        ds_label = int(ds_labels_matching[i])

        # originally labeled no_relation_index
        if ds_label == no_relation_index:
            if chosen_label == no_relation_index:
                norel_kept_batch += 1
            else:
                assert (chosen_label == ignore_index)
                norel_ignored_batch += 1

        # originally not labeled no_relation_index
        else:
            if chosen_label == ds_label:  # label is kept
                ds_kept_batch += 1
            elif chosen_label == no_relation_index:  # label is changed to other_class
                ds_to_norel_batch += 1
            else:
                assert (chosen_label == ignore_index)
                ds_ignored_batch += 1

    return norel_kept_batch, norel_ignored_batch, ds_kept_batch, ds_to_norel_batch, ds_ignored_batch


def smooth_collect_stats(collect_stats, tuple_index, smoothing_length):
    # smooth statistics for plots
    smoothed_stats = [sum(tpl[tuple_index] for tpl in collect_stats[idx:idx + smoothing_length]) / (
        sum(map(sum, collect_stats[idx:(idx + smoothing_length)]))) for idx in
                      range(len(collect_stats) - smoothing_length + 1)]

    return smoothed_stats


def eval_gold(chosen_labels, weak_label, gold_labels, mislabeled, ignore_index=-100, other_class=None):
    # comparison with gold labels
    if other_class is None:

        correctly_ignored = len(
            np.intersect1d(np.where(chosen_labels == ignore_index)[0], np.where(mislabeled == 1)[0]))
        falsely_ignored = len(np.intersect1d(np.where(chosen_labels == ignore_index)[0], np.where(mislabeled == 0)[0]))
        correctly_kept = len(np.intersect1d(np.where(chosen_labels != ignore_index)[0], np.where(mislabeled == 0)[0]))
        falsely_kept = len(np.intersect1d(np.where(chosen_labels != ignore_index)[0], np.where(mislabeled == 1)[0]))

        assert correctly_ignored + falsely_ignored + correctly_kept + falsely_kept == len(chosen_labels)
        return correctly_ignored, falsely_ignored, correctly_kept, falsely_kept

    else:
        correctly_ignored_ds = len(np.intersect1d(np.where(chosen_labels == ignore_index)[0],
                                                  np.where((mislabeled == 1) & (weak_label != other_class))[0]))
        correctly_ignored_other = len(np.intersect1d(np.where(chosen_labels == ignore_index)[0],
                                                     np.where((mislabeled == 1) & (weak_label == other_class))[0]))

        falsely_ignored_ds = len(np.intersect1d(np.where(chosen_labels == ignore_index)[0],
                                                np.where((mislabeled == 0) & (weak_label != other_class))[0]))
        falsely_ignored_other = len(np.intersect1d(np.where(chosen_labels == ignore_index)[0],
                                                   np.where((mislabeled == 0) & (weak_label == other_class))[0]))

        correctly_kept_ds = len(
            np.intersect1d(np.where((chosen_labels != ignore_index) & (chosen_labels != other_class))[0],
                           np.where((mislabeled == 0) & (weak_label != other_class))[0]))
        correctly_kept_other = len(np.intersect1d(np.where(chosen_labels != ignore_index)[0],
                                                  np.where((mislabeled == 0) & (weak_label == other_class))[0]))

        falsely_kept_ds = len(
            np.intersect1d(np.where((chosen_labels != ignore_index) & (chosen_labels != other_class))[0],
                           np.where((mislabeled == 1) & (weak_label != other_class))[0]))
        falsely_kept_other = len(np.intersect1d(np.where(chosen_labels != ignore_index)[0],
                                                np.where((mislabeled == 1) & (weak_label == other_class))[0]))

        correctly_corrected = len(np.intersect1d(np.where(chosen_labels == other_class)[0],
                                                 np.where((mislabeled == 1) & (gold_labels == other_class))[0]))
        falsely_corrected = len(np.intersect1d(np.where(chosen_labels == other_class)[0],
                                               np.where((weak_label != other_class) & (gold_labels != other_class))[0]))

        # check if values sum up correctly
        assert correctly_ignored_ds + correctly_ignored_other + falsely_ignored_ds + falsely_ignored_other + \
               correctly_kept_ds + correctly_kept_other + falsely_kept_ds + falsely_kept_other + correctly_corrected + \
               falsely_corrected == len(chosen_labels)

        return correctly_ignored_ds, correctly_ignored_other, falsely_ignored_ds, falsely_ignored_other, \
               correctly_kept_ds, correctly_kept_other, falsely_kept_ds, falsely_kept_other, correctly_corrected, \
               falsely_corrected


def make_plots(collect_stats, smoothing_length=1, title: str = None, storing_loc=None) -> None:
    if title:
        plt.title(title)

    # if other is not None:
    plt.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
    plt.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
    plt.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
    plt.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)
    plt.plot(smooth_collect_stats(collect_stats, 4, smoothing_length), linewidth=2)

    plt.gca().set_ylim([0, 1])
    plt.xlabel('Updates')

    if storing_loc is not None:
        plt.savefig(storing_loc,
                    bbox_inches='tight')
    plt.show()


def make_plots_legend(collect_stats, smoothing_length=1, storing_loc=None) -> None:
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(111)
    line_1, = ax.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
    line_2, = ax.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
    line_3, = ax.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
    line_4, = ax.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)
    line_5, = ax.plot(smooth_collect_stats(collect_stats, 4, smoothing_length), linewidth=2)
    legendFig.legend([line_1, line_2, line_3, line_4, line_5],
                     ["Label 0 Kept", "Label 0 Ignored", "Label 1 Kept", "Label 1 to Label 0", "Label 1 Ignored"],
                     loc='center', fontsize=30)

    if storing_loc is not None:
        plt.savefig(storing_loc, bbox_inches='tight')
    plt.show()


def make_plots_gold(collect_stats, smoothing_length=1, title: str = None, other_class=None, reduction=True,
                    storing_loc=None) -> None:
    plt.figure(1)

    if title:
        plt.title(title)

    if other_class is None:
        plt.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)

        plt.gca().set_ylim([0, 1])
        plt.xlabel('Updates')

        if storing_loc is not None:
            plt.savefig(storing_loc, bbox_inches='tight')

        plt.show()

    elif reduction is False:
        plt.plot(smooth_collect_stats(collect_stats, 0, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 1, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 2, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 3, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 4, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 5, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 6, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 7, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 8, smoothing_length))
        plt.plot(smooth_collect_stats(collect_stats, 9, smoothing_length))

        plt.gca().set_ylim([0, 1])
        plt.xlabel('Updates')

        if storing_loc is not None:
            plt.savefig(
                storing_loc, bbox_inches='tight')
        plt.show()

    else:
        # correctly_ignored_ds, correctly_ignored_other, falsely_ignored_ds, falsely_ignored_other, correctly_kept_ds,
        # correctly_kept_other, falsely_kept_ds, falsely_kept_other, correctly_corrected, falsely_corrected
        reduced_stats = [(tpl[0] + tpl[1], tpl[2] + tpl[3], tpl[4] + tpl[5], tpl[6] + tpl[7], tpl[8], tpl[9]) for tpl in
                         collect_stats]
        plt.plot(smooth_collect_stats(reduced_stats, 0, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(reduced_stats, 1, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(reduced_stats, 2, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(reduced_stats, 3, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(reduced_stats, 4, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(reduced_stats, 5, smoothing_length), linewidth=2)

        plt.gca().set_ylim([0, 1])
        plt.xlabel('Updates')

        if storing_loc is not None:
            plt.savefig(
                storing_loc, bbox_inches='tight')

        plt.show()


def make_plots_gold_legend(collect_stats, smoothing_length=1, other_class=None, storing_loc=None):
    if other_class is None:
        fig = plt.figure("Line plot")
        legendFig = plt.figure("Legend plot")
        ax = fig.add_subplot(111)
        line_1, = ax.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
        line_2, = ax.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
        line_3, = ax.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
        line_4, = ax.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)

        legendFig.legend([line_1, line_2, line_3, line_4],
                         ["Correctly Ignored", "Falsely Ignored", "Correctly Kept", "Falsely Kept"],
                         loc='center', fontsize=30)
        if storing_loc is not None:
            plt.savefig(
                storing_loc, bbox_inches='tight')
        plt.show()

    else:
        reduced_stats = [
            (tpl[0] + tpl[1], tpl[2] + tpl[3], tpl[4] + tpl[5], tpl[6] + tpl[7], tpl[8], tpl[9]) for tpl in collect_stats
        ]

        fig = plt.figure("Line plot")
        legendFig = plt.figure("Legend plot")
        ax = fig.add_subplot(111)
        line_1, = ax.plot(smooth_collect_stats(reduced_stats, 0, smoothing_length), linewidth=2)
        line_2, = ax.plot(smooth_collect_stats(reduced_stats, 1, smoothing_length), linewidth=2)
        line_3, = ax.plot(smooth_collect_stats(reduced_stats, 2, smoothing_length), linewidth=2)
        line_4, = ax.plot(smooth_collect_stats(reduced_stats, 3, smoothing_length), linewidth=2)
        line_5, = ax.plot(smooth_collect_stats(reduced_stats, 4, smoothing_length), linewidth=2)
        line_6, = ax.plot(smooth_collect_stats(reduced_stats, 5, smoothing_length), linewidth=2)

        legendFig.legend([line_1, line_2, line_3, line_4, line_5, line_6],
                         ["Correctly Ignored", "Falsely Ignored", "Correctly Kept", "Falsely Kept",
                          "Correctly Corrected", "Falsely Corrected"],
                         loc='center', fontsize=30)

        if storing_loc is not None:
            plt.savefig(storing_loc, bbox_inches='tight')

        plt.show()
