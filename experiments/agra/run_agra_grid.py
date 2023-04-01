import argparse
import itertools
import os
import statistics
import logging

import numpy as np
import torch
from scipy import stats
from torch import Tensor

from experiments.agra.utils import AGRAImageDataSet, load_image_dataset, define_data_encoding_agra
from experiments.utils import define_eval_metric, get_mv_train_labels, get_cifar_data, \
    load_train_labels_from_file
from src.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from src.utils import set_seed, compute_weights
from wrench.dataset import load_dataset, BaseDataset


logger = logging.getLogger(__name__)


# hyper-parameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)
thresholds = [0, 0.05, 0.01, 0.001]
all_hyperparams_combinations = list(itertools.product(batch_sizes, learning_rates, weight_decay, thresholds))
num_epochs = 10

# number of experiments will be performed with different initializations for the same setting in grid search
num_grid_exp = 3
# number of final experiments to be performed with the best parameter combination
num_final_exp = 5

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_train_data_for_agra(
        dataset, data_path, train_labels_path: str = None, num_valid_samples: int = None, finetuning_batch_size: int = 32,
        enc_model: str = "resnet50", finetuning: bool = False, finetuning_epochs: int = 2, metric: str = "acc"
):
    if dataset in ['youtube', 'sms', 'trec', 'yoruba', 'hausa']:
        # load wrench dataset
        train_data, test_data, valid_data = load_dataset(
            data_path, dataset, dataset_type='TextDataset', extract_fn="tfidf", extract_feature=True
        )
        # calculate train labels y_train with majority vote
        train_labels = get_mv_train_labels(train_data)
        train_data.labels = train_labels

    elif dataset in ['cifar', 'chexpert']:

        # load datasets
        train_data, test_data, valid_data, y_valid, y_test = get_cifar_data(
            os.path.join(data_path, dataset), num_valid_samples)

        # upload the labels from the file
        train_labels_dict = load_train_labels_from_file(data_path, train_labels_path, dataset)
        y_train = np.array(list(train_labels_dict.values()))

        num_classes = max(int(max(y_train)), int(max(y_valid)), int(max(y_test))) + 1

        # load Cifar and CheXpert datasets and get encodings with resnet-50
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_image_dataset(
            data_path, dataset, train_data, test_data, valid_data, enc_model,
            num_classes=num_classes, finetuning=finetuning, finetuning_epochs=finetuning_epochs, metric=metric,
            batch_size=finetuning_batch_size
        )

        # transform the data into wrench-compatible datasets
        train_data = AGRAImageDataSet(Tensor(train_features), Tensor(train_labels))
        valid_data = AGRAImageDataSet(Tensor(valid_features), Tensor(valid_labels))
        test_data = AGRAImageDataSet(Tensor(test_features), Tensor(test_labels))

    else:
        raise ValueError(f"Dataset {dataset} is not yet supported.")

    return train_data, valid_data, test_data, train_labels


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
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--finetuning_epochs", type=int, default=2)
    parser.add_argument("--finetuning_batch_size", type=int, default=2)
    parser.add_argument("--modification", type=str, default="last")
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--train_loss", type=str, default=None, choices=['CE', 'F1'])
    parser.add_argument("--closs", type=str, default=None, choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default=None, choices=['True', 'False'])
    parser.add_argument('--save', type=bool, default=True)

    parser.add_argument('--include_bias', type=str)
    parser.add_argument('--adaptive_threshold', type=str)

    args = parser.parse_args()
    args.include_bias = True if args.include_bias == "True" else False
    args.adaptive_threshold = True if args.adaptive_threshold == "True" else False

    logger.info(args)

    # set the seed
    if args.seed is not None:
        logger.info(f"Seed {args.seed} is fixed.")
        set_seed(args.seed)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding_agra(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(
        args.output_path, "results", 'grid_search', 'agra', args.dataset,
        f"results_bias_{args.include_bias}_thresh_{args.adaptive_threshold}")
    os.makedirs(results_folder, exist_ok=True)
    logger.info(f"Results will be solved to the following directory: {results_folder}")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load dataset and training labels (mv or separately provided)
    train_dataset, valid_dataset, test_dataset, train_labels = load_train_data_for_agra(
        args.dataset, dataset_path, args.train_labels_path, args.num_valid_samples, args.finetuning_batch_size, encoding,
        finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs, metric=metric
    )

    # calculate num_classes
    if args.other is not None:
        num_classes = int(max(args.other, max(train_labels), max(valid_dataset.labels))) + 1
    else:
        num_classes = int(max(max(train_labels), max(valid_dataset.labels))) + 1

    weights = compute_weights(train_labels)

    if args.train_loss is None and args.closs is None and args.weights is None:
        logger.info("All possible combinations for AGRA will be tried for this dataset")
        unique_combinations = [('CE', 'CE', 'False'), ('CE', 'F1', 'False'), ('CE', 'CE', 'True'), ('CE', 'F1', 'True')]

        for curr_train_loss, curr_comp_loss, curr_use_weights in unique_combinations:
            f = os.path.join(results_folder, f"trl_{curr_train_loss}_cml_{curr_comp_loss}_weights_{curr_use_weights}.txt")

            # compute weights for comparison batch sampling
            agra_weights = weights if curr_use_weights == "True" else None

            best_comb, best_metric = None, 0
            for param_comb in all_hyperparams_combinations:
                batch_size, lr, weight_decay, thresh = param_comb

                setup = f"Current setup: train_loss={curr_train_loss}, comp_loss={curr_comp_loss}, " \
                        f"use_weights={curr_use_weights}, batch_size={batch_size}, lr={lr}, " \
                        f"weight_decay={weight_decay}, thresh={thresh} \n"
                logger.info(setup)
                with open(f, "a+") as out_f:
                    out_f.write(setup)

                all_metrics = []
                # perform num_exp runs for each combination
                for i in range(num_grid_exp):
                    seed = np.random.randint(0, 1000)
                    # initialize the model
                    model = LogRegModelWithAGRA(
                        num_classes=num_classes,
                        agra_weights=agra_weights,
                        other=args.other,
                        agra_threshold=thresh,

                        include_bias=args.include_bias,
                        adaptive_threshold=args.include_bias
                    )

                    model.fit(
                        dataset_train=train_dataset,
                        dataset_valid=valid_dataset,

                        lr=lr,
                        l2=weight_decay,
                        comp_loss=curr_comp_loss,
                        batch_size=batch_size,
                        num_epochs=num_epochs,

                        modification=args.modification,
                        metric=metric,
                        verbose=True
                    )

                    # test the trained model on the test set
                    valid_metric_value = model.test(valid_dataset, batch_size, metric)
                    all_metrics.append(valid_metric_value)
                    logger.info(f"Experiment: {i}, {metric}: {valid_metric_value}")

                avg_metric = np.average(all_metrics)
                curr_res = f"Combination: {param_comb} \t exp metrics: {all_metrics} \t average: {avg_metric} \n"
                logger.info(curr_res)
                with open(f, "a+") as out_f:
                    out_f.write(curr_res)

                if avg_metric > best_metric:
                    best_metric = avg_metric
                    best_comb = param_comb
                    logger.info(f"New best parameter combi: {param_comb}")
                logger.info("------------------------------")

            # what are our parameters for the final run?
            best_batch_size, best_lr, best_weight_decay, best_thresh = best_comb
            with open(f, "a+") as out_f:
                out_f.write(f"The best parameter combination for the setting curr_train_loss={curr_train_loss}, "
                            f"curr_comp_loss={curr_comp_loss}, curr_use_weights={curr_use_weights} is: "
                            f"best_batch_size={best_batch_size}, best_lr={best_lr}, "
                            f"best_weight_decay={best_weight_decay}, best_thresh={best_thresh} \n")

            # final run
            all_metric_best_combi = []
            for i in range(num_final_exp):
                seed = np.random.randint(0, 1000)
                # initialize the model
                model = LogRegModelWithAGRA(
                    num_classes=num_classes,
                    agra_weights=agra_weights,
                    other=args.other,
                    agra_threshold=best_thresh,

                    include_bias=args.include_bias,
                    adaptive_threshold=args.include_bias
                )

                model.fit(
                    dataset_train=train_dataset,
                    dataset_valid=valid_dataset,

                    lr=best_lr,
                    l2=best_weight_decay,
                    comp_loss=curr_comp_loss,
                    batch_size=best_batch_size,
                    num_epochs=num_epochs,

                    modification=args.modification,
                    metric=metric,
                    verbose=True
                )
                # test the trained model on the test set
                test_metric_value = model.test(test_dataset, best_batch_size, metric)
                all_metric_best_combi.append(test_metric_value)

            res_line = f"Best parameter combi: batch_size={best_batch_size}, lr={best_lr}, " \
                       f"weight_decay={best_weight_decay}, thresh={best_thresh} \n " \
                       f"All metrics: {all_metric_best_combi} \n " \
                       f"Average: {np.average(all_metric_best_combi)}, " \
                       f"std: {round(statistics.stdev(all_metric_best_combi), 5)}, " \
                       f"sem: {round(stats.sem(all_metric_best_combi), 5)} \n\n"
            logger.info(res_line)
            with open(f, "a+") as out_f:
                out_f.write(res_line)
