import copy
import logging
import os
import time

import numpy as np
import math
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.multi_label.eval import make_plots, eval_performance, collect_statistics, make_legend
from src.multi_label.labels_recalculation_binary_multilabel_no_bias import calculate_label, F1LossMulti, MaskedBCELoss
from src.multi_label.logistic_regression import MaxEntNetwork
from src.multi_label.utils_multilabel import set_seed, AGRADataset

logger = logging.getLogger(__name__)


def train_grad_match_multilabel(
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        num_features: int = 1280,
        output_classes: int = 13,  # 12 pathologies + "No Finding" (1 if all 12 pathologies 0; 0 else)
        batch_size: int = 128,  # originally 64
        learning_rate: int = 0.001,
        metric: str = "F1",
        num_epochs: int = 50,
        seed: int = 0,
        weight_decay=0,
        training_loss="BCE",
        comparison_loss="BCE",
        loss_reduction: str = None,
        metric_avg: str = "macro",
        ignore_index=-100,
        thresh: int = 0,
        storing_loc: str = None,
        print_plots: str = False
):
    """
    Args:
        path_to_embeddings: path to the folder "train_embeddings"
        path_to_embeddings_dev: path to the folder "valid_embeddings"
        number_of_embeddings: number of embeddings in the folder "train_embeddings"
        number_of_embeddings_dev: number of embeddings in the folder "valid_embeddings"
        num_features: number of input features for the model
        output_classes: number of output classes
        batch_size: batch size for the DataLoaders
        num_epochs: number of epochs that the model should train for
        seed: seed that should be used for making results reproducible
        storing_loc: path to the location where the model should be stored if save_model is True
    Returns:
        best performance on the validation set and after which epoch it was achieved
    """

    if not (metric == "AUROC"):
        # todo: implement other metrics
        raise ValueError("Currently only AUROC metric is supported to conduct experiments with CheXpert dataset.")

    # collect epoch times
    epoch_times = {}

    # set seed
    set_seed(seed)

    # initialize network
    net = MaxEntNetwork(num_features, output_classes)

    # determine the number of training samples
    number_of_train_samples = X_train.shape[0]

    # set loss
    if training_loss == "BCE":
        criterion = MaskedBCELoss(ignore_index, output_classes, reduction=loss_reduction)

    elif training_loss == "F1":
        criterion = F1LossMulti(ignore_index, reduction=metric_avg)

    # set loss for comparison
    if comparison_loss == "BCE":
        comparison_criterion = nn.BCELoss()
        loss_type = "mean"

    elif comparison_loss == "F1":
        comparison_criterion = F1LossMulti(ignore_index=None,
                                           reduction=metric_avg)
        loss_type = "sum"

    # load data
    train_data_ds = AGRADataset(X_train, y_train)
    dev_data = AGRADataset(X_dev, y_dev)
    test_data = AGRADataset(X_test, y_test)

    # initialize train dataloader
    train_dataloader_ds_with_subs = DataLoader(train_data_ds, batch_size=batch_size, shuffle=True)

    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # keep track of statistics and performance
    running_loss = 0.0
    best_auroc = 0  # initialize high val loss
    best_epoch = 1
    statistics = []
    steps = 0

    for epoch in range(num_epochs):

        # start time for epoch
        start_time = time.time()

        for batch, (inputs_ds_with_sub, labels_ds_with_sub) in enumerate(train_dataloader_ds_with_subs, start=1):

            steps += 1

            # sample a comparison batch
            idx = list(
                WeightedRandomSampler(np.ones(number_of_train_samples), num_samples=batch_size, replacement=False)
            )
            train_dataloader_orig = DataLoader(torch.utils.data.Subset(train_data_ds, idx), batch_size=batch_size)
            (inputs_ds_orig, labels_ds_orig) = next(iter(train_dataloader_orig))

            chosen_labels = calculate_label(
                inputs_ds_orig,
                labels_ds_orig,
                copy.deepcopy(net),
                optimizer,
                comparison_criterion,
                inputs_ds_with_sub,
                labels_ds_with_sub,
                num_classes=output_classes,
                ignore_index=ignore_index,
                loss_type=loss_type,
                threshold=thresh
            )

            # count number of discarded labels (per class)
            # statistics.append(collect_statistics(chosen_labels.shape[0], output_classes, ignore_index, chosen_labels,
                                                 # labels_ds_with_sub))

            # compute masked loss
            optimizer.zero_grad()
            output_ds = torch.sigmoid(net(inputs_ds_with_sub))
            loss = criterion(output_ds, chosen_labels)
            running_loss += loss

            # perform the update with the chosen labels
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                logger.info(f"[{epoch + 1}, {batch}] loss: {running_loss / steps:.3f}")
                running_loss = 0.0
                steps = 0

        # evaluate the performance on the dev data
        auroc = eval_performance(net, dev_data, metric_avg=metric_avg)

        if auroc >= best_auroc:
            best_auroc = auroc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(net.state_dict())

        end = time.time()
        epoch_times[epoch] = time.time() - start_time
        logger.info(f"Epoch: {str(epoch + 1)}, took: {str(end - start_time)}")

    logger.info(f"Avg epoch time: {sum(epoch_times.values()) / len(epoch_times)}")
    logger.info(f"Best epoch: {best_epoch}")

    # load best performing model and evaluate it on the test set
    net_AGRA = MaxEntNetwork(num_features, output_classes)
    net_AGRA.load_state_dict(best_model_state)

    if storing_loc is not None:
        torch.save({
            'model_state_dict': net_AGRA.state_dict(),
            # 'statistics': statistics
        }, os.path.join(storing_loc + ".pt"))

    # evaluate model on dev and test set
    val_auroc, dev_probs = eval_performance(net_AGRA, dev_data, metric_avg=metric_avg, return_preds=True)

    test_auroc, test_probs = eval_performance(net_AGRA, test_data, metric_avg=metric_avg, return_preds=True)

    # label_0_kept, label_0_ignored, label_1_kept, label_1_ignored = [sum(x) for x in zip(
    #     *statistics[0:(math.ceil(number_of_train_samples / batch_size) * best_epoch)])]
    # logger.info(
    #     f'label_0_kept: {label_0_kept}, label_0_ignored: {label_0_ignored}, label_1_kept: {label_1_kept}, '
    #     f'label_1_ignored: {label_1_ignored} '
    # )
    # logger.info(
    #     f'label_0_kept%: {label_0_kept / (label_0_kept + label_0_ignored)}, '
    #     f'label_0_ignored%: {label_0_ignored / (label_0_kept + label_0_ignored)}, '
    #     f'label_1_kept%: {label_1_kept / (label_1_kept + label_1_ignored)}, '
    #     f'label_1_ignored:% {label_1_ignored / (label_1_kept + label_1_ignored)}'
    # )
    #
    # ignored_0 = label_0_ignored / (label_0_kept + label_0_ignored + label_1_kept + label_1_ignored)
    # ignored_1 = label_1_ignored / (label_0_kept + label_0_ignored + label_1_kept + label_1_ignored)
    #
    # kept_0 = label_0_kept / (label_0_kept + label_0_ignored + label_1_kept + label_1_ignored)
    # kept_1 = label_1_kept / (label_0_kept + label_0_ignored + label_1_kept + label_1_ignored)

    # plot the statistics
    if print_plots:
        make_plots(statistics, smoothing_length=10)
        make_legend(statistics, smoothing_length=10)

    return val_auroc, test_auroc, 0, 0, 0, 0, dev_probs, test_probs # val_auroc, test_auroc, ignored_0, ignored_1, kept_0, kept_1, dev_probs, test_probs
