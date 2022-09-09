import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def collect_statistics(
        batch_size: int, num_labels: int, ignore_index: int, chosen_labels: np.ndarray, original_labels: np.ndarray
):
    negative_label_kept, negative_label_ignored, positive_label_kept, positive_label_ignored = 0, 0, 0, 0
    for i in range(batch_size):
        for j in range(num_labels):

            chosen_label = chosen_labels[i, j]
            ds_label = original_labels[i, j]

            if torch.equal(ds_label, torch.tensor(0).float()):
                if torch.equal(chosen_label, ds_label):
                    negative_label_kept += 1
                else:
                    assert (ignore_index in chosen_label)
                    negative_label_ignored += 1
            else:
                assert (torch.equal(ds_label, torch.tensor(1).float()))
                if torch.equal(chosen_label, torch.tensor(1).float()):
                    positive_label_kept += 1
                else:
                    assert (ignore_index in chosen_label)
                    positive_label_ignored += 1

    # check if stats sum up correctly
    assert (sum([negative_label_kept, negative_label_ignored, positive_label_kept,
                 positive_label_ignored]) == batch_size * num_labels)

    return negative_label_kept, negative_label_ignored, positive_label_kept, positive_label_ignored


def make_plots(collect_stats, smoothing_length=100, batch_size=64,
               title: str = "Ignored Samples During Training") -> None:
    if title:
        plt.title(title)
    for i in range(0, len(collect_stats[0])):
        plt.plot(smooth_collect_stats(collect_stats, i, smoothing_length, batch_size), linewidth=2)
    plt.gca().set_ylim([0, 1])
    plt.xlabel('Updates')
    plt.savefig(
        r"C:\Users\Lena\Documents\Data Science\Masterarbeit Paper\new_chexpert_plot\train_plot_BCE.png",
        bbox_inches='tight')
    plt.show()


def make_legend(collect_stats, batch_size=1, smoothing_length=10):
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(111)

    line_1, = ax.plot(smooth_collect_stats(collect_stats, 0, smoothing_length, batch_size), linewidth=2)
    line_2, = ax.plot(smooth_collect_stats(collect_stats, 1, smoothing_length, batch_size), linewidth=2)
    line_3, = ax.plot(smooth_collect_stats(collect_stats, 2, smoothing_length, batch_size), linewidth=2)
    line_4, = ax.plot(smooth_collect_stats(collect_stats, 3, smoothing_length, batch_size), linewidth=2)

    legendFig.legend([line_1, line_2, line_3, line_4],
                     ["Label 0 Kept", "Label 0 Ignored", "Label 1 Kept", "Label 1 Ignored"],
                     loc='center', fontsize=30)
    plt.show()


def smooth_collect_stats(collect_stats, tuple_index, smoothing_length, batch_size):
    smoothed_stats = [sum(tpl[tuple_index] for tpl in collect_stats[idx:idx + smoothing_length]) / (
        sum(map(sum, collect_stats[idx:(idx + smoothing_length)])))
                      for idx in range(len(collect_stats) - smoothing_length + 1)]
    return smoothed_stats


def eval_performance(net, dataset, return_preds=False):
    prediction_probs = []
    labels = []

    # get predictions
    with torch.no_grad():
        for i, (data, target) in enumerate(dataset):
            outputs = torch.sigmoid(net(data))
            labels.append(target.detach().numpy())
            prediction_probs.append(outputs.detach().numpy())

    # turn labels and predictions to arrays
    labels = np.asarray(labels)
    prediction_probs = np.asarray(prediction_probs)

    # remove the labels for Lung Lesion, Pleural Other and Fracture
    labels = np.delete(labels, [3, 10, 11], 1)
    prediction_probs = np.delete(prediction_probs, [3, 10, 11], 1)

    # compute AUROC
    roc_auc = roc_auc_score(labels, prediction_probs, average=None)
    mean_auc = np.mean(roc_auc)

    print("AUROC per class:", roc_auc)
    print("Mean AUROC", mean_auc, "\n")

    if return_preds is False:
        return mean_auc
    else:
        return mean_auc, prediction_probs
