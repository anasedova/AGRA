import argparse
import logging
import os
import statistics

import numpy as np
from scipy import stats

from depr.src import train_grad_match_multilabel
from depr.src.multi_label.utils_multilabel import set_seed, load_data

logger = logging.getLogger(__name__)

epochs = 5

# hyper-parameters for grid search; values from WRENCH
batch_sizes = [32, 64, 128]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)
threshold = [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]

# number of experiments will be performed with different initializations for the same setting in grid search
num_grid_exp = 3
# number of final experiments to be performed with the best parameter combination
num_final_exp = 5

train_loss = "F1"
comp_loss = "F1"
metric_avg = "macro"


class AGRAExperiments:

    def __init__(self,
                 input_data_path: str,
                 output_data_path: str,
                 metric: str = "AUROC",
                 seed: int = None
                 ):

        self.input_data_path = input_data_path
        self.output_data_path = os.path.join(output_data_path)
        self.metric = metric

        if seed:
            set_seed(int(seed))

        os.makedirs(self.output_data_path, exist_ok=True)

        # load data from csv files
        self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test = load_data(self.input_data_path)

    def run_experiments(self):
        # run AGRA experiments
        combs, mean_dev_performances_grid = self.grid_search()
        chosen_comb = combs[np.argmax(mean_dev_performances_grid)]  # extract the chosen values of the grid search
        dev_performances, test_performances = self.final(chosen_comb)  # run final experiments

        logger.info(f"All dev {self.metric}: {dev_performances} \n All test performances: {test_performances}")
        logger.info(
            f"Mean {self.metric} on dev set: {round(sum(dev_performances) / len(dev_performances), 5)}, "
            f"std: {round(statistics.stdev(dev_performances), 5)}, sem: {round(stats.sem(dev_performances), 5)}")
        logger.info(
            f"Mean {self.metric} on test set: {round(sum(test_performances) / len(test_performances), 5)}, "
            f"std: {round(statistics.stdev(test_performances), 5)}, sem: {round(stats.sem(test_performances), 5)}")

    def grid_search(self):
        combs, mean_dev_performances_grid = [], []
        for batch_size in batch_sizes:
            for thresh in threshold:
                for learning_rate in learning_rates:
                    for decay in weight_decay:

                        # keep track of achieved performances
                        dev_performances = []

                        combs.append((learning_rate, batch_size, decay, thresh))
                        logger.info(f"Current combination: {(learning_rate, batch_size, decay, thresh)}")

                        # perform num_exp runs for each combination
                        for _ in range(num_grid_exp):
                            seed = np.random.randint(0, 1000)

                            val_metric, test_metric, ignored_0, ignored_1, kept_0, kept_1, dev_probs, test_probs = \
                                train_grad_match_multilabel(
                                    self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test,
                                    seed=seed,
                                    training_loss=train_loss,
                                    comparison_loss=comp_loss,
                                    metric_avg=metric_avg,
                                    num_features=1280,
                                    output_classes=12,
                                    num_epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    metric=self.metric,
                                    weight_decay=decay,
                                    loss_reduction="avg_per_class",  # for BCE loss
                                    thresh=thresh)

                            dev_performances.append(val_metric)

                        # compute mean performance over 3 runs for the hyperparameter combination
                        mean_dev_performances_grid.append(np.mean(dev_performances))

        return combs, mean_dev_performances_grid

    def final(self, chosen_comb):

        dev_performances, test_performances = [], []

        learning_rate = chosen_comb[0]
        batch_size = chosen_comb[1]
        decay = chosen_comb[2]
        thresh = chosen_comb[3]

        logger.info(f"Chosen Learning Rate: {learning_rate}")
        logger.info(f"Chosen Batch Size: {batch_size}")
        logger.info(f"Chosen Weight Decay: {decay}")
        logger.info(f"Chosen Removal Threshold: {thresh}")

        for i in range(num_final_exp):
            seed = np.random.randint(0, 1000)
            set_seed(seed)
            storing_loc_models = self.output_data_path if self.output_data_path else None

            val_metric, test_metric, ignored_0, ignored_1, kept_0, kept_1, dev_probs, test_probs = \
                train_grad_match_multilabel(
                    self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test,
                    seed=seed,
                    training_loss=train_loss,
                    comparison_loss=comp_loss,
                    num_features=1280,
                    output_classes=12,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=decay,
                    loss_reduction="avg_per_class",  # for BCE loss
                    thresh=thresh,
                    storing_loc=storing_loc_models
                )

            dev_performances.append(val_metric)
            test_performances.append(test_metric)

        return dev_performances, test_performances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', default='', type=str)
    parser.add_argument('--output_data_path', default='', type=str)
    parser.add_argument('--metric', default='', type=str)
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    logger.info(args)

    exp = AGRAExperiments(
        input_data_path=args.path_to_data,
        output_data_path=args.output_data_path,
        metric=args.metric,
        seed=args.seed
    )
    exp.run_experiments()
