import logging
import os
import argparse
import statistics
import itertools
import time

import numpy as np
from scipy import stats
import warnings
from itertools import permutations

from src.single_label.gradient_matching_single_label_no_bias_fixed_with_bert import train_grad_match_with_gold
from src.single_label.utils import set_seed, load_model
from src.single_label.wrench_dataload import prepare_data_bert

logger = logging.getLogger(__name__)

epochs = 10  # number of epochs
ignore_index = -100  # class index assigned to ignored samples

# hyper-parameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)

# number of experiments will be performed with different initializations for the same setting in grid search
num_grid_exp = 3
# number of final experiments to be performed with the best parameter combination
num_final_exp = 5


class AGRAExperiments:

    def __init__(
            self,
            input_data_path: str,
            dataset: str,
            output_data_path: str,
            metric: str = "acc",
            metric_avg: str = None,
            crit: str = None,
            train_loss: str = "CE",
            comp_loss: str = "CE",
            use_weights: bool = False,
            seed: int = None,
            other: int = None
    ):
        self.dataset = dataset
        self.metric = metric
        self.train_loss = train_loss
        self.comp_loss = comp_loss
        self.use_weights = use_weights
        self.other = other

        if seed:
            set_seed(seed)

        self.storing_loc = os.path.join(
            output_data_path, self.dataset, f"trl_{self.train_loss}_cml_{self.comp_loss}_weights_{str(self.use_weights)}"
        )
        os.makedirs(self.storing_loc, exist_ok=True)


        self.train_data, self.train_labels_ds, self.dev_data, self.dev_labels, self.test_data, \
        self.test_labels, self.gold_labels, self.mislabeled, weights, self.noise_rate = \
            prepare_data_bert(dataset, input_data_path)

        self.sample_weights = weights if use_weights else None

        # which averaging will be used for AGRA update (when we optimizing towards the F1 score)
        if metric_avg is None:  # default: binary if num_classes=2, macro otherwise
            self.metric_avg = "binary" if max(self.gold_labels) == 1 else "macro"
        else:
            self.metric_avg = metric_avg

        # which averaging will be used for evaluation score calculation
        if crit is None:  # default: binary if num_classes=2, macro otherwise
            self.crit = "binary" if max(self.gold_labels) == 1 else "macro"
        else:
            self.crit = crit

        self.input_ags_validity_check()

    def run_experiments(self):
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

        # save performance:
        with open(os.path.join(self.storing_loc, 'performance.txt'), 'w') as f:
            f.write(f"Mean {self.metric} on dev set: {round(sum(dev_performances) / len(dev_performances), 5)}, \n")
            f.write(f"std: {round(statistics.stdev(dev_performances), 5)}, sem: {round(stats.sem(dev_performances), 5)}, \n")
            f.write(f"Mean {self.metric} on test set: {round(sum(test_performances) / len(test_performances), 5)}, \n")
            f.write(f"std: {round(statistics.stdev(test_performances), 5)}, sem: {round(stats.sem(test_performances), 5)}, \n")
            f.write("chosen params: " + str(chosen_comb))

    def grid_search(self):
        combs, mean_dev_performances_grid = [], []
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for decay in weight_decay:

                    # keep track of achieved performances
                    dev_performances = []

                    # keep track of tried combinations
                    combs.append((learning_rate, batch_size, decay))
                    logger.info("========================================")
                    logger.info(f"Current combination: {(learning_rate, batch_size, decay)}")
                    logger.info("========================================")

                    # perform num_exp runs for each combination
                    for _ in range(num_grid_exp):

                        seed = np.random.randint(0, 1000)

                        # train gradient matching with specified parameters
                        acc, f1, precision, recall, acc_test, f1_test, precision_test, recall_test, ignored, \
                        ignored_mislabeled, ignored_correctly_labeled, mislabeled_ignored, correctly_corrected, \
                        corrected, test_predictions = train_grad_match_with_gold(
                                self.dataset,
                                self.train_data,
                                self.train_labels_ds,
                                self.dev_data,
                                self.dev_labels,
                                self.test_data,
                                self.test_labels,
                                gold_labels=self.gold_labels,
                                mislabeled=self.mislabeled,
                                other_class=self.other,
                                seed=seed,
                                num_epochs=epochs,
                                batch_size=batch_size,
                                weight_decay=decay,
                                comp_batch_size=batch_size,
                                learning_rate=learning_rate,
                                training_loss=self.train_loss,
                                comparison_loss=self.comp_loss,
                                sample_weights=self.sample_weights,
                                metric=self.metric,
                                metric_avg=self.metric_avg,
                                crit=self.crit,
                                threshold=0
                            )

                        # record dev performance
                        if self.metric == "acc":
                            dev_performances.append(acc)

                        if self.metric == "F1":
                            dev_performances.append(f1)

                    # average performance for parameter setting over num_grid_exp runs
                    mean_dev_performances_grid.append(np.mean(dev_performances))
                    logger.info(f"Average dev performance for this set up: {np.mean(dev_performances)}")

        return combs, mean_dev_performances_grid

    def final(self, chosen_comb):

        dev_performances, test_performances = [], []

        learning_rate = chosen_comb[0]
        batch_size = chosen_comb[1]
        decay = chosen_comb[2]

        logger.info(f"Chosen Learning Rate: {learning_rate}")
        logger.info(f"Chosen Batch Size: {batch_size}")
        logger.info(f"Chosen Weight Decay: {decay}")

        for i in range(num_final_exp):
            seed = np.random.randint(0, 1000)
            set_seed(seed)
            storing_loc_models = self.storing_loc if self.storing_loc else None

            # train gradient matching with specified parameters
            acc, f1, precision, recall, acc_test, f1_test, precision_test, recall_test, ignored, ignored_mislabeled, \
            ignored_correctly_labeled, mislabeled_ignored, correctly_corrected, corrected, test_predictions = \
                train_grad_match_with_gold(
                    self.train_sentences_ds,
                    self.train_labels_ds,
                    self.dev_sentences,
                    self.dev_labels,
                    self.test_sentences,
                    self.test_labels,
                    gold_labels=self.gold_labels,
                    mislabeled=self.mislabeled,
                    other_class=self.other,
                    seed=seed,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    weight_decay=decay,
                    comp_batch_size=batch_size,
                    learning_rate=learning_rate,
                    training_loss=self.train_loss,
                    comparison_loss=self.comp_loss,
                    sample_weights=self.sample_weights,
                    metric=self.metric,
                    metric_avg=self.metric_avg,
                    crit=self.crit,
                    storing_loc=storing_loc_models,
                    threshold=0
                )

            if self.metric == "acc":
                dev_performances.append(acc)
                test_performances.append(acc_test)

            if self.metric == "F1":
                dev_performances.append(f1)
                test_performances.append(f1_test)

            print(
                {"acc": acc, "f1": f1, "precision": precision, "recall": recall, "acc_test": acc_test,
                 "f1_test": f1_test, "precision_test": precision_test, "recall_test": recall_test, "ignored": ignored,
                 "ignored_mislabeled": ignored_mislabeled, "ignored_correctly_labeled": ignored_correctly_labeled,
                 "mislabeled_ignored": mislabeled_ignored, "correctly_corrected": correctly_corrected,
                 "corrected": corrected, "test_predictions": test_predictions}
            )

        return dev_performances, test_performances

    def input_ags_validity_check(self):

        if self.train_loss not in ["CE", "F1"] or self.comp_loss not in ["CE", "F1", None]:
            raise ValueError(
                "Unacceptable loss value! Losses implemented so far are cross-entropy (CE) or F1-Score (F1)."
            )

        if self.metric_avg is not None and self.metric_avg not in ["binary", "macro"]:
            raise ValueError(
                "Wrong metric averaging value. Accepted metric averaging are either binary ('binary') "
                "or macro ('macro')."
            )
        if self.metric_avg is not None and (self.comp_loss == "CE" and self.train_loss == "CE"):
            warnings.warn(f"No averaging will be done since CE losses will be used everywhere.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--other', default=None)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--weights', default='False')
    args = parser.parse_args()
    logger.info(args)

    file_path = os.path.split(os.path.abspath(__file__))[0]
    path_to_data = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')
    output_path = os.path.join(file_path, 'results_no_bias_fixed')

    use_weights = True if args.weights == "True" else False
    seed = int(args.seed) if args.seed else None
    other = int(args.other) if args.other else None

    datasets = ['youtube', 'sms', 'trec', 'yoruba', 'hausa']

    for dataset in datasets:
        print('Current dataset:', dataset)

        if dataset in ['youtube', 'trec']:
            metric = 'acc'
        else:
            metric = 'F1'
            logger.info("All possible combinations for AGRA will be tried for this dataset")
            unique_combinations = [('CE', 'CE', 'False'), ('CE', 'F1', 'False'), ('CE', 'CE', 'True'), ('CE', 'F1', 'True')]

            for train_loss, comp_loss, weights in unique_combinations:
                print('Current combination:', train_loss, comp_loss, use_weights)
                f = os.path.join(output_path, dataset, f"trl_{train_loss}_cml_{comp_loss}_weights_{use_weights}")
                os.makedirs(f, exist_ok=True)

                use_weights = True if weights == "True" else False

                exp = AGRAExperiments(
                        input_data_path=path_to_data,
                        dataset=dataset,
                        output_data_path=output_path,
                        metric=metric,
                        metric_avg=None,
                        crit=None,
                        train_loss=train_loss,
                        comp_loss=comp_loss,
                        use_weights=use_weights,
                        seed=seed,
                        other=other
                )
                exp.run_experiments()
# todo: time
