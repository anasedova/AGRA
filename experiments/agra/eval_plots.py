import matplotlib.pyplot as plt
import os
import torch
import numpy as np

# adjust plot frames
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def get_statistics(weak_labels: torch.tensor, chosen_labels: torch.tensor, gold_labels: torch.tensor, other: int, ignore_index: int):
    """
    computes label change statistics
    :param chosen_labels: tensor of weak labels for the batch
    :param chosen_labels: tensor of labels for the batch chosen by AGRA
    :param gold_labels: tensor of gold labels for the batch
    :param other: alternative label
    :return: statistics on the label changes
    """

    # no alternative label
    if other is None:
        # Correctly labeled and kept
        correctly_labeled = torch.where(weak_labels == gold_labels)[0]
        kept = torch.where(chosen_labels != ignore_index)[0]
        correctly_kept = len(np.intersect1d(correctly_labeled, kept))

        # Correctly labeled and removed
        removed = torch.where(chosen_labels == ignore_index)[0]
        falsely_removed = len(np.intersect1d(correctly_labeled, removed))

        # Mislabeled and kept
        mislabeled = torch.where(weak_labels != gold_labels)[0]
        falsely_kept = len(np.intersect1d(mislabeled, kept))

        # Mislabeled and removed
        correctly_removed = len(np.intersect1d(mislabeled, removed))

        correctly_corrected = 0 # since no other class
        falsely_corrected = 0

    else:
        correctly_labeled = torch.where(weak_labels == gold_labels)[0]
        mislabeled = torch.where(weak_labels != gold_labels)[0]
        assert len(correctly_labeled) + len(mislabeled) == weak_labels.shape[0]

        corrected = torch.where((weak_labels != other) & (chosen_labels == other))[0]
        removed = torch.where(chosen_labels == ignore_index)[0]
        kept = np.where(weak_labels == chosen_labels)[0]
        assert len(corrected) + len(removed) + len(kept) == weak_labels.shape[0]

        gold_other_class = torch.where(gold_labels == other)[0]
        false_other_class = torch.where(gold_labels != other)[0]

        correctly_kept = len(np.intersect1d(correctly_labeled, kept))
        falsely_removed = len(np.intersect1d(correctly_labeled, removed))
        falsely_kept = len(np.intersect1d(mislabeled, kept))
        correctly_removed = len(np.intersect1d(mislabeled, removed))

        correctly_corrected = len(np.intersect1d(corrected, gold_other_class))
        falsely_corrected = len(np.intersect1d(corrected, false_other_class))

    assert correctly_kept + falsely_removed + falsely_kept + correctly_removed + correctly_corrected + falsely_corrected == len(weak_labels)

    return correctly_removed, falsely_removed, falsely_kept, correctly_kept, correctly_corrected, falsely_corrected

def smooth_collect_stats(collect_stats, tuple_index, smoothing_length):
    """
    smooth statistics for plots
    :param collect_stats: all stats collected throughout the run
    :param tuple_index: current element
    :param smoothing_length: determines the degree of smoothing
    :return: smoothed stats
    """
    smoothed_stats = [(sum(tpl[tuple_index] for tpl in collect_stats[idx:idx + smoothing_length]) / (sum(map(sum, collect_stats[idx:(idx + smoothing_length)]))))*100 for idx in range(len(collect_stats) - smoothing_length + 1)]

    return smoothed_stats

def make_plots_gold(collect_stats, smoothing_length=5, other: int=None, title: str=None, storing_loc=None) -> None:
    """
    plots the statistics and saves the plot
    :param collect_stats: all stats collected throughout the run
    :param smoothing_length: determines the degree of smoothing
    :param other: alternative label
    :param title: optional title string
    :param storing_loc: storing location
    :return:
    """

    if title:
        plt.title(title)

    if other is None: # without other class

        plt.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)

        # check if stats sum up correctly
        # sum_per_batch = [a+b+c+d for a,b,c,d in zip(smooth_collect_stats(collect_stats, 0, smoothing_length), smooth_collect_stats(collect_stats, 1, smoothing_length), smooth_collect_stats(collect_stats, 2, smoothing_length), smooth_collect_stats(collect_stats, 3, smoothing_length))]

        plt.gca().set_ylim([0, 70])
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
        plt.xlabel('Update')
        plt.ylabel('Percent of Samples')

        # "Correctly Removed", "Falsely Removed", "Correctly Kept", "Falsely Kept"
        plt.legend(["Mislabeled Removed", "Correctly Labeled Removed", "Mislabeled Kept", "Correctly Labeled Kept"], ncol=2, prop={'size':'small'}, loc="upper center")

        if storing_loc is not None:
            plt.savefig(os.path.join(storing_loc, 'plot.png'), bbox_inches='tight')

        plt.show()

    else:  # with other class
        # correctly_ignored_ds, correctly_ignored_other, falsely_ignored_ds, falsely_ignored_other, correctly_kept_ds, correctly_kept_other, falsely_kept_ds, falsely_kept_other, correctly_corrected, falsely_corrected
        plt.plot(smooth_collect_stats(collect_stats, 0, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 1, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 2, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 3, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 4, smoothing_length), linewidth=2)
        plt.plot(smooth_collect_stats(collect_stats, 5, smoothing_length), linewidth=2)

        plt.gca().set_ylim([0, 100])
        plt.xlabel('Update')

        plt.legend(["Mislabeled Removed", "Correctly Labeled Removed", "Correctly Labeled Kept", "Mislabeled Kept", "Correctly Corrected", "Falsely Corrected"])

        if storing_loc is not None:
            plt.savefig(os.path.join(storing_loc, 'plot.png'), bbox_inches='tight')

        plt.show()

# def make_plots_threshold(performances, title: str=None, storing_loc=None) -> None:
#     thresholds = [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
#     if title:
#         plt.title(title)
#     for perf in performances:
#         line = [p*100 for p in perf]
#         plt.plot(thresholds, line)
#     plt.gca().set_ylim([0, 100])
#     plt.xlabel(r'Removal Threshold $\tau$')
#     plt.ylabel(r'Test Performance')
#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
#     plt.legend(["Youtube", "SMS", "TREC", "Yorùbá", "Hausa"])
#     plt.xticks(thresholds)
#     if storing_loc is not None:
#         plt.savefig(os.path.join(storing_loc, "threshold_plot.png"), bbox_inches='tight')
#     plt.show()