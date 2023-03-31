import copy
import math
import time
import logging

import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim, nn
from wrench.utils import get_bert_torch_dataset_class, construct_collate_fn_trunc_pad
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.single_label.eval import (
    collect_statistics, make_plots, make_plots_legend, make_plots_gold_legend, eval_grad_match, eval_gold,
    make_plots_gold
)
from src.single_label.labels_recalculation_no_bias_fixed import calculate_label, F1Loss
from src.single_label.utils import get_bert_model, set_seed
from transformers import AutoTokenizer
from src.single_label.bert_classifier import BertTextClassifier

logger = logging.getLogger(__name__)
collate_fn = construct_collate_fn_trunc_pad('mask')

def train_grad_match_with_gold(
        dataset: str,
        train_data,
        train_y_ds: np.ndarray,
        dev_data,
        dev_y: np.ndarray,
        test_data,
        test_y: np.ndarray,
        gold_labels: torch.tensor = None,
        mislabeled: torch.tensor = None,
        other_class: int = None,
        output_classes: int = None,
        training_loss: str = "CE",
        comparison_loss: str = "CE",
        batch_size: int = 128,
        comp_batch_size: int = 128,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        num_epochs: int = 3,
        seed: int = 0,
        ignore_index: int = -100,
        metric: str = "F1",
        sample_weights: np.ndarray = None,
        metric_avg: str = "macro",
        crit: str = "macro",
        threshold: float = 0,
        print_plots: bool = False,
        smooth: int = 1,
        storing_loc: str = None,
        storing_loc_plot_train: str = None,
        storing_loc_plot_legend: str = None,
        storing_loc_plot_gold: str = None,
        storing_loc_plot_gold_legend: str = None
):

    # empty dictionary for epoch times
    epoch_times = {}

    # set seed for reproducibility
    set_seed(seed)

    # determine the number of features and samples in the dataset
    num_samples = train_data.__len__()

    # determine the number of classes if not given as input
    if output_classes is None and other_class is not None:
        output_classes = max(max(train_y_ds), max(dev_y), other_class) + 1
        output_classes = int(output_classes)

    elif output_classes is None and other_class is None:
        output_classes = max(max(train_y_ds), max(dev_y)) + 1
        output_classes = int(output_classes)

    # initialize the neural network
    net, model_name = get_bert_model(dataset=dataset, output_classes=output_classes)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dataset = get_bert_torch_dataset_class(train_data)(train_data, tokenizer, 512)

    train_dataloader_ds_with_subs = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn)

    # todo: continue adaption here (and correct mistakes above)

    # initialize criterion for the update
    if training_loss == "CE":
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif training_loss == "F1":
        criterion = F1Loss(num_classes=output_classes, avg=metric_avg)

    # initialize criterion for comparison
    if comparison_loss == "CE":
        comparison_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss_type = "mean" # for autograd

    elif comparison_loss == "F1":
        comparison_criterion = F1Loss(num_classes=output_classes, avg=metric_avg)
        loss_type = "sum" # for autograd

    # initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # keep track of stats during training
    running_loss = 0.0
    steps = 0
    statistics = []
    statistics_gold = []

    # initialize best metric and epoch
    best_accuracy = 0
    best_f1 = 0
    best_epoch_f1 = 1
    best_epoch_accuracy = 1
    for epoch in range(num_epochs):

        # keep track of runtime
        start = time.time()

        # sample a batch from the training set
        for batch, (inputs_ds_with_sub, labels_ds_with_sub, gold_labels, mislabeled) in \
                enumerate(train_dataloader_ds_with_subs, start=1):

            steps += 1

            # sample batch from the comparison dataset
            # if no weights specified -> sample uniformly (all datapoints have the same weight)
            if sample_weights is None:
                idx = list(WeightedRandomSampler(
                    np.ones(num_samples), num_samples=int(comp_batch_size), replacement=False)
                )

            # if sample weights are specified use a weighted sampler
            else:
                idx = list(WeightedRandomSampler(sample_weights, num_samples=int(comp_batch_size), replacement=False))

            # establish a dataloader and retrieve the comparison batch from it
            train_dataloader_orig = DataLoader(
                torch.utils.data.Subset(train_data, idx), batch_size=int(comp_batch_size)
            )
            (inputs_ds_orig, labels_ds_orig, _, _) = next(iter(train_dataloader_orig))

            # choose a label for each datapoint in batch (ws label, other_class or ignore)
            chosen_labels = calculate_label(
                inputs_ds_with_sub,
                labels_ds_with_sub,
                inputs_ds_orig,
                labels_ds_orig,
                copy.deepcopy(net),
                criterion,
                comparison_criterion,
                ignore_index=ignore_index,
                other_class=other_class,
                threshold=threshold
            )

            # collect statistics for the current batch
            if other_class is not None:
                grouping = other_class
            elif other_class is None and output_classes == 2:
                # even if no "other class", separate how many samples with label 0 and 1 are kept/ignored in the case
                # of binary classification
                grouping = 0
            else:
                grouping = None

            # norel_kept_batch, norel_ignored_batch, ds_kept_batch, ds_to_norel_batch, ds_ignored_batch
            stats = collect_statistics(
                inputs_ds_with_sub.shape[0], ignore_index, chosen_labels, labels_ds_with_sub,
                no_relation_index=grouping
            )

            # correctly_ignored, falsely_ignored, correctly_kept, falsely_kept (if no other class)
            # correctly_ignored_ds, correctly_ignored_other, falsely_ignored_ds, falsely_ignored_other,
            # correctly_kept_ds, correctly_kept_other, falsely_kept_ds, falsely_kept_other, correctly_corrected,
            # falsely_corrected
            stats_gold = eval_gold(
                chosen_labels, labels_ds_with_sub, gold_labels, mislabeled, ignore_index, other_class
            )

            # assert if numbers of gold comparison align with statistics
            if other_class:
                # all "other" that are kept (correctly or incorrectly)
                assert stats[0] == (stats_gold[5] + stats_gold[7])
                # all "other" that are ignored (correctly or incorrectly)
                assert stats[1] == (stats_gold[1] + stats_gold[3])
                # all ws that are kept (correctly or incorrectly)
                assert stats[2] == (stats_gold[4] + stats_gold[6])
                # all ws labels that are changed
                assert stats[3] == (stats_gold[8] + stats_gold[9])
                # all ws that are ignored (correctly or incorrectly)
                assert stats[4] == (stats_gold[0] + stats_gold[2])

            statistics.append(stats)
            statistics_gold.append(stats_gold)

            # remove ignored samples
            ignore_samples = np.where(chosen_labels == ignore_index)
            chosen_labels = np.delete(chosen_labels, ignore_samples)

            # Finally do optimization with DS data and chosen labels
            optimizer.zero_grad()

            output_ds = net(inputs_ds_with_sub)

            # remove outputs for ignored samples -> make sure they are not used for updating
            remove_mask = np.zeros([len(output_ds)])
            remove_mask[ignore_samples] = 1
            output_ds = output_ds[remove_mask != 1]

            # perform the update if the number of kept samples is > 0
            if len(output_ds) > 0:
                loss = criterion(output_ds, chosen_labels)

                # stop execution if loss is NaN -> likely an error in loss function
                if math.isnan(loss):
                    raise RuntimeError("Loss NaN")

                running_loss += loss.item()
                loss.backward()

                optimizer.step()

            # if all samples ignored -> don't update, reduce number of steps trained by 1
            # (so that averaging is not affected)
            else:
                steps -= 1

            # print the average loss every few batches
            # if batch % 10 == 0:
            #     if steps > 0:
            #         logger.info(f"[{epoch + 1}, {batch}] loss: {running_loss / steps:.3f}")
            #     else:
            #         logger.info("No updates made")
            #
            #     # reset running loss and steps
            #     running_loss = 0.0
            #     steps = 0

        # record epoch time
        epoch_times[epoch] = time.time() - start

        # evaluate the model on the dev set
        accuracy, f1, _, _ = eval_grad_match(dev_X, dev_y, net, avg=crit)

        # record the best performance and in which epoch it was achieved
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch_accuracy = epoch + 1
            best_model_state_accuracy = copy.deepcopy(net.state_dict())

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch_f1 = epoch + 1
            best_model_state_f1 = copy.deepcopy(net.state_dict())

        # end = time.time()

        # logger.info(f"Epoch {str(epoch + 1)}, took: {str(end - start)}")

    # if evaluation metric is F1 -> choose best epoch accordingly
    net_AGRA = get_model(model, dataset, num_features, output_classes)

    if metric == "F1":
        chosen_epoch = best_epoch_f1
        net_AGRA.load_state_dict(best_model_state_f1)

    # if evaluation metric is Accuracy -> choose best epoch accordingly
    elif metric == "acc":
        chosen_epoch = best_epoch_accuracy
        net_AGRA.load_state_dict(best_model_state_accuracy)

    else:
        raise ValueError(f"The metric {metric} is not supported")

    # cut off statistics after chosen epoch
    statistics = statistics[0:(math.ceil(num_samples / batch_size) * chosen_epoch)]
    statistics_gold = statistics_gold[0:(math.ceil(num_samples / batch_size) * chosen_epoch)]

    if storing_loc is not None:
        torch.save({
            'model_state_dict': net_AGRA.state_dict(),
            'statistics': statistics,
            'statistics_gold': statistics_gold
        }, os.path.join(storing_loc + ".pt"))

    # compare label changes to gold labels
    if other_class is None:

        correctly_ignored, falsely_ignored, correctly_kept, falsely_kept = [sum(x) for x in zip(*statistics_gold)]

        # compute fraction of mislabeled samples that were ignored
        if (correctly_ignored + falsely_kept) > 0:
            ignored_mislabeled = correctly_ignored/(correctly_ignored + falsely_kept)

        else:
            ignored_mislabeled = 0

        # compute fraction of ignored samples that were mislabeled
        if (correctly_ignored + falsely_ignored) > 0:
            mislabeled_ignored = correctly_ignored / (correctly_ignored + falsely_ignored)

        else:
            mislabeled_ignored = 0

        # compute fraction of correctly labeled samples that were ignored
        if (falsely_ignored + correctly_kept) > 0: # all correctly labeled
            ignored_correctly_labeled = falsely_ignored/(falsely_ignored + correctly_kept)

        else:
            ignored_correctly_labeled = 0

        # compute total percentage of ignored samples
        ignored = (falsely_ignored + correctly_ignored)/\
                  (falsely_ignored + correctly_ignored + falsely_kept + correctly_kept)

        # no samples corrected in this case
        corrected = 0
        correctly_c = 0

    # other class label included
    else:
        correctly_ignored_ds, correctly_ignored_other, falsely_ignored_ds, falsely_ignored_other, correctly_kept_ds, \
        correctly_kept_other, falsely_kept_ds, falsely_kept_other, correctly_corrected, falsely_corrected = \
            [sum(x) for x in zip(*statistics_gold)]

        if (correctly_ignored_ds + falsely_kept_ds + correctly_ignored_other + falsely_kept_other + correctly_corrected) > 0: # all mislabeled samples
            ignored_mislabeled = (correctly_ignored_ds + correctly_ignored_other)/\
                                 (correctly_ignored_ds + falsely_kept_ds + correctly_ignored_other + falsely_kept_other
                                  + correctly_corrected)

        else:
            ignored_mislabeled = 0

        if (falsely_ignored_ds + correctly_kept_ds + falsely_ignored_other + correctly_kept_other + falsely_corrected) > 0: # all correctly labeled samples
            ignored_correctly_labeled = (falsely_ignored_ds + falsely_ignored_other)/(falsely_ignored_ds +
                                                                                      correctly_kept_ds +
                                                                                      falsely_ignored_other +
                                                                                      correctly_kept_other +
                                                                                      falsely_corrected)

        else:
            ignored_correctly_labeled = 0

        # compute fraction of ignored samples that were mislabeled
        if (correctly_ignored_ds + falsely_ignored_ds + correctly_ignored_other + falsely_ignored_other) > 0: # all ignored samples
            mislabeled_ignored = (correctly_ignored_ds + correctly_ignored_other) / (correctly_ignored_ds +
                                                                                     falsely_ignored_ds +
                                                                                     correctly_ignored_other +
                                                                                     falsely_ignored_other)

        else:
            mislabeled_ignored = 0

        # compute fraction of warranted label changes
        if (correctly_corrected + falsely_corrected) > 0:
                correctly_c = correctly_corrected / (correctly_corrected + falsely_corrected)

        else:
            correctly_c = 0

        # compute total percentage of ignored samples
        ignored = (correctly_ignored_ds + falsely_ignored_ds + falsely_ignored_other + correctly_ignored_other) / \
                  (correctly_ignored_ds + falsely_ignored_ds + correctly_ignored_other + falsely_ignored_other +
                   correctly_kept_ds + falsely_kept_ds + correctly_kept_other + falsely_kept_other +
                   correctly_corrected + falsely_corrected)

        # compute total percentage of corrected samples
        corrected = (correctly_corrected + falsely_corrected) / \
                    (correctly_ignored_ds + falsely_ignored_ds + correctly_ignored_other + falsely_ignored_other +
                     correctly_kept_ds + falsely_kept_ds + correctly_kept_other + falsely_kept_other +
                     correctly_corrected + falsely_corrected)

    # logger.info(f"Avg epoch time: {sum(epoch_times.values()) / len(epoch_times)}")

    # logger.info(f"Best Epoch F1: {best_epoch_f1}")
    # logger.info(f"Best Epoch Accuracy: {best_epoch_accuracy}")

    # evaluate the model on the dev set
    accuracy, f1, precision, recall = eval_grad_match(dev_X, dev_y, net_AGRA, avg=crit)

    # evaluate the model on the test set
    accuracy_test, f1_test, precision_test, recall_test, test_predictions = eval_grad_match(
        test_X, test_y, net_AGRA, avg=crit, return_preds=True
    )

    # print performance
    logger.info(f"Dev. Accuracy: {accuracy}, Dev. F1({str(crit)}): {f1}")
    logger.info(f"Test Accuracy: {accuracy_test}, Test F1({str(crit)}): {f1_test}")

    # plot the statistics over the training process
    if print_plots is True:
        make_plots(
            statistics, title="Label Changes during Training", smoothing_length=smooth,
            storing_loc=storing_loc_plot_train
        )
        make_plots_legend(statistics, smoothing_length=smooth, storing_loc=storing_loc_plot_legend)
        make_plots_gold(
            statistics_gold, smoothing_length=smooth, title="Comparison with Gold Labels", other_class=other_class,
            storing_loc=storing_loc_plot_gold
        )
        make_plots_gold_legend(
            statistics_gold, smoothing_length=smooth, other_class=other_class, storing_loc=storing_loc_plot_gold_legend
        )

    return accuracy, f1, precision, recall, accuracy_test, f1_test, precision_test, recall_test, ignored, \
           ignored_mislabeled, ignored_correctly_labeled, mislabeled_ignored, correctly_c, corrected, test_predictions
