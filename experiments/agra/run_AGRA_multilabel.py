import argparse
import logging
import os
import statistics

import numpy as np
import torch

from experiments.agra.utils import load_train_data_for_agra, define_data_encoding_agra
from experiments.utils import define_eval_metric
from src.AGRA.logreg_model_with_AGRA_multilabel import LogRegModelWithAGRAMulti
from src.utils import set_seed

logger = logging.getLogger(__name__)

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = 5  # number of epochs

# hyperparameters for grid search
# hyperparameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)

num_grid_exp = 3
num_final_exp = 5

### to reproduce CheXpert, set:
# batch_sizes = [128]
# learning_rates = [0.001]
# weight_decay = [0.001]

# num_grid_exp = 1
# num_final_exp = 5

set_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help='path to the dataseet folder', type=str, default=None)
    parser.add_argument("--output_path", help='path to results folder', type=str, default="./results")
    parser.add_argument("--include_bias",
                        help='True: include the bias gradients in the similarity computation (to reprocude ECML set to False for chexpert)',
                        type=bool, default=False)
    parser.add_argument("--closs",
                        help='comparison loss function',
                        choices=['CE', 'F1'],
                        type=str, default='F1')
    parser.add_argument("--dataset", type=str, help='dataset that should be run', default='chexpert')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--save', type=bool, default=True)

    # The following arguments are only relevant for experiments on CIFAR-10; they do not affect the reproduction of the CheXpert results
    parser.add_argument("--train_labels_path", help='path to the noisy training labels (only required for CIFAR-10)',
                        type=str, default=None)
    parser.add_argument("--gold_label_path", help='path to the god training labels (only required for CIFAR-10)',
                        type=str, default=None)
    parser.add_argument("--num_valid_samples",
                        help='number of validation instances that should be sampled from test set if no valid set is available (only required for CIFAR-10)',
                        type=int, default=5000)
    parser.add_argument("--finetuning_epochs", help='Number of fine-tuning epochs if finetuining is True', type=int,
                        default=100)
    parser.add_argument("--finetuning_batch_size", help='Batch size that is used for fine-tuning', type=int, default=64)

    args = parser.parse_args()
    logger.info(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 'datasets')

    # define the datasets that should be run
    encoding = define_data_encoding_agra(args)

    print('Current dataset:', args.dataset, "\n")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # set the seed
    if args.seed is not None:
        set_seed(args.seed)

        # load dataset
        train_dataset, valid_dataset, test_dataset, train_labels_gold, train_labels_noisy = load_train_data_for_agra(
                    args.dataset, dataset_path, args.train_labels_path, args.num_valid_samples, args.finetuning_batch_size,
                    encoding, finetuning_epochs=args.finetuning_epochs
        )

        num_classes = train_labels_noisy.shape[1]

        print('Current comparison loss:', args.closs)

        # storing location for results
        results_folder = os.path.join(args.output_path, "results", 'agra', args.dataset, f"trl_{'CE'}_cml_{args.closs}")
        os.makedirs(results_folder, exist_ok=True)
        output_file = open(os.path.join(results_folder, "test_performance.txt"), "w+")

        # perform grid search
        combs, mean_dev_performances_grid = [], []
        print('Starting grid search.')
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for decay in weight_decay:

                    # keep track of achieved performances
                    dev_performances = []

                    # keep track of tried combinations
                    combs.append((lr, batch_size, decay))
                    print(f"Current hyperparameter combination: {(lr, batch_size, decay)}")

                    # perform num_exp runs for each combination
                    for _ in range(num_grid_exp):
                        seed = np.random.randint(0, 1000)

                        model = LogRegModelWithAGRAMulti(
                            num_classes=num_classes,
                            agra_threshold=0,
                            include_bias=args.include_bias
                        )

                        val_performance = model.fit(
                            dataset_train=train_dataset,
                            dataset_valid=valid_dataset,
                            y_train=train_labels_noisy,
                            comp_loss=args.closs,
                            lr=lr,
                            l2=decay,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            metric=metric,
                            verbose=True,
                            seed=seed
                        )
                        dev_performances.append(val_performance)
                    # average performance for parameter setting over num_grid_exp runs
                    mean_dev_performances_grid.append(np.mean(dev_performances))
                    print(f"Average dev performance: {np.mean(dev_performances)}")

        # after grid search is completed, choose setting
        chosen_comb = combs[np.argmax(mean_dev_performances_grid)]

        # run model with chosen parameter combination and record test performance
        test_performances = []

        lr = chosen_comb[0]
        batch_size = chosen_comb[1]
        decay = chosen_comb[2]

        print("\n")
        print(f"Learning Rate: {lr}")
        print(f"Batch Size: {batch_size}")
        print(f"Weight Decay: {decay}", "\n")

        print('Starting final model runs with above parameters.')
        for i in range(num_final_exp):
            seed = np.random.randint(0, 1000)

            model = LogRegModelWithAGRAMulti(
                    num_classes=num_classes,
                    agra_threshold=0,
                    include_bias=args.include_bias
                )

            val_performance = model.fit(
                    dataset_train=train_dataset,
                    dataset_valid=valid_dataset,
                    y_train=train_labels_noisy,
                    comp_loss=args.closs,
                    lr=lr,
                    l2=decay,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    metric=metric,
                    verbose=True,
                    seed=seed
            )

            test_metric_value = model.test_chexpert(test_dataset, batch_size, metric, test_mode=True)
            print('Run', str(i + 1), 'performance:', test_metric_value)
            test_performances.append(test_metric_value)

        print("\n")
        print('Mean test performance:', sum(test_performances) / len(test_performances))
        print('Std. of test performance:', statistics.stdev(test_performances), "\n")

        output_file.write("AGRA with LogReg \n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["chosen lr", str(lr)]) + "\n")
        output_file.write("\t".join(["chosen batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["chosen weight decay", str(decay)]) + "\n")
        output_file.write("\t".join(['test performance (mean)', str(sum(test_performances) / len(test_performances))]) + "\n")
        output_file.write("\t".join(['test performance (std)', str(statistics.stdev(test_performances))]))




