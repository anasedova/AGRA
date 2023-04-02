import argparse
import itertools
import logging
import os
import statistics

import numpy as np
import torch
from scipy import stats
from torch import Tensor

from experiments.agra.utils import AGRAImageDataSet, load_image_dataset, define_data_encoding_agra, \
    load_train_data_for_agra
from experiments.baselines.feature_based.logreg_model_baseline_wrench import LogRegBaseline
from experiments.utils import define_eval_metric, get_mv_train_labels, get_cifar_data, \
    load_train_labels_from_file
from src.utils import set_seed
from wrench.dataset import load_dataset, BaseDataset

logger = logging.getLogger(__name__)

# hyper-parameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)
all_hyperparams_combinations = list(itertools.product(batch_sizes, learning_rates, weight_decay))
num_epochs = 10

# number of experiments will be performed with different initializations for the same setting in grid search
num_grid_exp = 3
# number of final experiments to be performed with the best parameter combination
num_final_exp = 5


# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_labels_path", type=str, default=None)
    parser.add_argument("--gold_label_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='youtube',
                        choices=['youtube', 'sms', 'trec', 'yoruba', 'hausa', 'cifar', 'chexpert'])
    parser.add_argument("--num_valid_samples", type=int, default=5000,
                        help="Size of a valid set to be sampled from the test set if no valid set is available")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--setting", type=str, default="noisy")
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--finetuning_epochs", type=int, default=2)
    parser.add_argument("--finetuning_batch_size", type=int, default=32)
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()

    # set the seed
    if args.seed is not None:
        set_seed(args.seed)

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'grid_search', 'gold', args.dataset)
    os.makedirs(results_folder, exist_ok=True)
    logger.info(f"Results will be solved to the following directory: {results_folder}")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding_agra(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'grid_search', 'gold', args.dataset)
    os.makedirs(results_folder, exist_ok=True)
    output_file = os.path.join(results_folder, "result.txt")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load dataset and training labels (mv or separately provided)
    train_dataset, valid_dataset, test_dataset, train_labels_gold, train_labels_noisy = \
        load_train_data_for_agra(
            args.dataset, dataset_path, args.train_labels_path, args.num_valid_samples, args.finetuning_batch_size,
            encoding, finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs, metric=metric
    )

    if args.setting == "gold":
        train_labels = train_labels_gold
    elif args.setting == "noisy":
        train_labels = train_labels_noisy
    else:
        raise ValueError("Unknown setting")

    # calculate num_classes
    if args.other is not None:
        num_classes = int(max(args.other, max(train_labels), max(valid_dataset.labels))) + 1
    else:
        num_classes = int(max(max(train_labels), max(valid_dataset.labels))) + 1

    best_comb, best_metric = None, 0
    for param_comb in all_hyperparams_combinations:
        batch_size, lr, weight_decay = param_comb
        setup = f"Current setup: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay} \n"
        logger.info(setup)
        with open(output_file, "a+") as out_f:
            out_f.write(setup)

        all_metrics = []
        # perform num_exp runs for each combination
        for i in range(num_grid_exp):
            seed = np.random.randint(0, 1000)

            model = LogRegBaseline(num_classes=num_classes, other=args.other)
            model.fit(
                train_dataset,
                y_train=train_labels,
                dataset_valid=valid_dataset,
                lr=lr,
                l2=weight_decay,
                batch_size=batch_size,
                num_epochs=num_epochs,
                metric=metric,
            )

            valid_metric_value = model.test(valid_dataset, batch_size, metric)
            all_metrics.append(valid_metric_value)
            logger.info(f"Experiment: {i}, {metric}: {valid_metric_value}")

        avg_metric = np.average(all_metrics)
        curr_res = f"Combination: {param_comb} \t exp metrics: {all_metrics} \t average: {avg_metric} \n"
        logger.info(curr_res)
        with open(output_file, "a+") as out_f:
            out_f.write(curr_res)

        if avg_metric > best_metric:
            best_metric = avg_metric
            best_comb = param_comb
            logger.info(f"New best parameter combi: {param_comb}")
        logger.info("------------------------------")

    # what are our parameters for the final run?
    best_batch_size, best_lr, best_weight_decay = best_comb
    with open(output_file, "a+") as out_f:
        out_f.write(f"The best parameter combination is: best_batch_size={best_batch_size}, best_lr={best_lr},"
                    f"best_weight_decay={best_weight_decay} \n")

    best_batch_size = 128
    best_lr = 0.01
    best_weight_decay = 1e-05

    # final run
    all_metric_best_combi = []
    for i in range(num_final_exp):
        seed = np.random.randint(0, 1000)

        model = LogRegBaseline(num_classes=num_classes, other=args.other)
        model.fit(
            train_dataset,
            y_train=train_labels,
            dataset_valid=valid_dataset,
            lr=best_lr,
            l2=best_weight_decay,
            batch_size=best_batch_size,
            num_epochs=num_epochs,
            metric=metric
        )

        test_metric_value = model.test(test_dataset, best_batch_size, metric)
        all_metric_best_combi.append(test_metric_value)

    res_line = f"Best parameter combi: batch_size={best_batch_size}, lr={best_lr}, " \
               f"weight_decay={best_weight_decay} \n " \
               f"All metrics: {all_metric_best_combi} \n " \
               f"Average: {np.average(all_metric_best_combi)}, " \
               f"std: {round(statistics.stdev(all_metric_best_combi), 5)}, " \
               f"sem: {round(stats.sem(all_metric_best_combi), 5)} \n\n"
    logger.info(res_line)

    with open(output_file, "a+") as out_f:
        out_f.write(res_line)
