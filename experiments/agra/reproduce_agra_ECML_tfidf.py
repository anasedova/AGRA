import argparse
import os
import logging
import torch
import statistics
import numpy as np

from experiments.agra.utils import load_train_data_for_agra
from experiments.utils import define_eval_metric
from src.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from src.utils import set_seed, compute_weights

logger = logging.getLogger(__name__)

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = 10  # number of epochs

# hyperparameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)

set_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help='path to the dataseet folder', type=str, default=None)
    parser.add_argument("--output_path", help='path to results folder', type=str, default="./results")
    parser.add_argument("--encoding", help='Data encoding, set to None for reproduction of ECML results', default=None,
                        choices=['tfidf', 'bert', 'resnet50'])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument("--include_bias", help='Whether or not to include the bias gradients in the similarity computation (to reprocude ECML set to False for text datasets)',
                        type=bool, default=False)

    # The following arguments are only relevant for experiments on CIFAR-10
    parser.add_argument("--train_labels_path", help='path to the noisy training labels (only required for CIFAR-10)', type=str, default=None)
    parser.add_argument("--gold_label_path", help='path to the god training labels (only required for CIFAR-10)',  type=str, default=None)
    parser.add_argument("--num_valid_samples", help='number of validation instances that should be sampled from test set if no valid set is available (only required for CIFAR-10)',
                        type=int, default=5000)
    parser.add_argument("--finetuning", help='If True fine-tunes a CNN to extract image embeddings (only required for CIFAR-10)', type=bool, default=False)
    parser.add_argument("--finetuning_epochs", help='Number of fine-tuning epochs if finetuining is True', type=int, default=2)
    parser.add_argument("--finetuning_batch_size", help='Batch size that is used for fine-tuning', type=int, default=2)

    # WIP, not used for ECML paper
    parser.add_argument("--adaptive_threshold", help='Adaptively determines threshold given the distribution of similarity scores (not used for ECML paper, DO NOT USE WITH OTHER CLASS)', type=bool, default=False)

    args = parser.parse_args()
    logger.info(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 'datasets')

    # define the datasets that should be run
    datasets = ['youtube', 'sms', 'trec', 'yoruba', 'hausa']

    # define the way how the dataset will be encoded
    encoding = "tfidf"

    for dataset in datasets:
        print('Current dataset:', dataset, "\n")
        # comparison loss and weighting options to try
        unique_combinations = [('CE', 'False'), ('F1', 'False'), ('CE', 'True'), ('F1', 'True')]

        # define metric for the evaluation
        metric = define_eval_metric(dataset)

        for closs, weights in unique_combinations:

            # load dataset
            train_dataset, valid_dataset, test_dataset, train_labels_gold, train_labels_noisy = load_train_data_for_agra(
                    dataset, dataset_path, args.train_labels_path, args.num_valid_samples, args.finetuning_batch_size,
                    encoding,
                    finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs, metric=metric
            )
            mislabeled = len(np.where(train_labels_gold != train_labels_noisy)[0])
            print('Noise rate:', mislabeled / len(train_labels_gold))

            num_classes = int(max(max(train_labels_noisy), max(valid_dataset.labels))) + 1

            print('Current combination:', (closs, weights))
            use_weights = True if weights == 'True' else False

            # define storing locations
            # for plots
            storing_loc = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'training_plots', dataset, f"comp_loss_{closs}_weights_{weights}")
            os.makedirs(storing_loc, exist_ok=True)

            # for results
            results_folder = os.path.join(args.output_path, "results_reproduce_seed", 'agra', dataset, f"trl_{'CE'}_cml_{closs}_weights_{weights}")
            os.makedirs(results_folder, exist_ok=True)
            output_file = open(os.path.join(results_folder, "test_performance.txt"), "w+")

            agra_weights = compute_weights(train_labels_noisy) if weights == 'True' else None

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
                        for _ in range(3):
                            seed = np.random.randint(0, 1000) # 684, 603, 671

                            model = LogRegModelWithAGRA(
                                num_classes=num_classes,
                                agra_weights=agra_weights,
                                other=None,
                                agra_threshold=0,
                                adaptive_threshold=args.adaptive_threshold,
                                storing_loc=storing_loc,
                                include_bias=args.include_bias
                            )

                            val_performance = model.fit(
                                dataset_train=train_dataset,
                                dataset_valid=valid_dataset,
                                y_train=train_labels_noisy,
                                comp_loss=closs,
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
            print(f"Chosen Learning Rate: {lr}")
            print(f"Chosen Batch Size: {batch_size}")
            print(f"Chosen Weight Decay: {decay}", "\n")

            print('Starting final model runs with chosen parameters.')
            for i in range(5):
                seed = np.random.randint(0, 1000)
                set_seed(seed)

                model = LogRegModelWithAGRA(
                    num_classes=num_classes,
                    agra_weights=agra_weights,
                    other=None,
                    agra_threshold=0,
                    adaptive_threshold=args.adaptive_threshold,
                    storing_loc=storing_loc,
                    include_bias=args.include_bias
                )

                val_performance = model.fit(
                    dataset_train=train_dataset,
                    dataset_valid=valid_dataset,
                    y_train=train_labels_noisy,
                    comp_loss=closs,
                    lr=lr,
                    l2=decay,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    metric=metric,
                    verbose=True
                )

                test_metric_value = model.test(test_dataset, batch_size, metric, test_mode=True)
                print('Run', str(i + 1), 'performance:', test_metric_value)
                test_performances.append(test_metric_value)

            print("\n")
            print('Mean test performance:', sum(test_performances) / len(test_performances))
            print('Std. of test performance:', statistics.stdev(test_performances), "\n")

            output_file.write("AGRA with LogReg \n")
            output_file.write("\t".join(["comp_loss", closs]) + "\n")
            output_file.write("\t".join(["weights", weights]) + "\n")
            output_file.write("\t".join(["alternative_label", 'none']) + "\n")
            output_file.write("\t".join(["chosen lr", str(lr)]) + "\n")
            output_file.write("\t".join(["chosen batch_size", str(batch_size)]) + "\n")
            output_file.write("\t".join(["chosen weight decay", str(decay)]) + "\n")
            output_file.write("\t".join(['test performance (mean)', str(sum(test_performances) / len(test_performances))]) + "\n")
            output_file.write("\t".join(['test performance (std)', str(statistics.stdev(test_performances))]))




