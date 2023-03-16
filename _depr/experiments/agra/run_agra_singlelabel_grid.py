import logging
import os
import argparse
import statistics
import warnings
import itertools

import numpy as np
from scipy import stats
from itertools import permutations

from depr.src.single_label.gradient_matching_single_label import train_grad_match_with_gold
from depr.src import set_seed, load_model
from depr.src.single_label.wrench_dataload import prepare_data

logger = logging.getLogger(__name__)

epochs = 10  # number of epochs
ignore_index = -100  # class index assigned to ignored samples

# hyper-parameters for grid search
batch_sizes = [32, 128, 512]
learning_rates = np.logspace(-5, -1, num=5, base=10)
weight_decay = np.logspace(-5, -1, num=5, base=10)
threshold = [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]

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
            extractor: str = "tfidf",
            model_name: str = None,
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
        self.extractor = extractor
        self.metric = metric
        self.train_loss = train_loss
        self.comp_loss = comp_loss
        self.use_weights = use_weights
        self.other = other

        if seed:
            set_seed(seed)

        self.storing_loc = os.path.join(
            output_data_path, self.dataset, f"trl_{self.train_loss}_cml_{self.comp_loss}"
        )
        os.makedirs(self.storing_loc, exist_ok=True)

        model_name = load_model(model_name) if extractor == "bert" else None
        dataset_type = "ImageDataset" if self.dataset == "cifar" else "TextDataset"

        self.train_sentences_ds, self.train_labels_ds, self.dev_sentences, self.dev_labels, self.test_sentences, \
        self.test_labels, self.gold_labels, self.mislabeled, weights, self.noise_rate = \
            prepare_data(dataset, dataset_type, input_data_path, extractor, model_name)

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

    def grid_search(self):
        combs, mean_dev_performances_grid = [], []
        for thresh in threshold:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    for decay in weight_decay:

                        # keep track of achieved performances
                        dev_performances = []

                        # keep track of tried combinations
                        combs.append((learning_rate, batch_size, decay, thresh))
                        logger.info("========================================")
                        logger.info(f"Current combination: {(learning_rate, batch_size, decay, thresh)}")
                        logger.info("========================================")

                        # perform num_exp runs for each combination
                        for _ in range(num_grid_exp):

                            seed = np.random.randint(0, 1000)

                            # train gradient matching with specified parameters
                            acc, f1, precision, recall, acc_test, f1_test, precision_test, recall_test, ignored, \
                            ignored_mislabeled, ignored_correctly_labeled, mislabeled_ignored, correctly_corrected, \
                            corrected, test_predictions = train_grad_match_with_gold(
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
                                threshold=thresh
                            )

                            # record dev performance
                            if self.metric == "Accuracy":
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
        thresh = chosen_comb[3]

        logger.info(f"Chosen Learning Rate: {learning_rate}")
        logger.info(f"Chosen Batch Size: {batch_size}")
        logger.info(f"Chosen Weight Decay: {decay}")
        logger.info(f"Chosen Removal Threshold: {thresh}")

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
                    threshold=thresh,
                    storing_loc=storing_loc_models
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
    parser.add_argument('--path_to_data', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--metric', default='', type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--extractor', default='tfidf', type=str)
    parser.add_argument('--train_loss', default=None, type=str)
    parser.add_argument('--comp_loss', default=None, type=str)
    parser.add_argument('--use_weights', default=True)
    parser.add_argument('--other', default=None)
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    logger.info(args)

    use_weights = True if args.use_weights == "True" else False
    seed = int(args.seed) if args.seed else None
    other = int(args.other) if args.other else None

    if args.train_loss is None and args.comp_loss is None:
        logger.info("All possible combinations for AGRA will be tried for this dataset")
        train_losses = ["F1", "CE"]
        comp_loss = ["F1", "CE"]
        unique_combinations = [list(zip(comb, comp_loss)) for comb in permutations(train_losses, len(comp_loss))]
        unique_combinations = list(itertools.chain.from_iterable(unique_combinations))

        for train_loss, comp_loss in unique_combinations:
            f = os.path.join(args.output_path, args.dataset, f"trl_{train_loss}_cml_{comp_loss}")
            os.makedirs(f, exist_ok=True)

            exp = AGRAExperiments(
                input_data_path=args.path_to_data,
                dataset=args.dataset,
                output_data_path=args.output_path,
                extractor=args.extractor,
                model_name=None,
                metric=args.metric,
                metric_avg=None,
                crit=None,
                train_loss=train_loss,
                comp_loss=comp_loss,
                use_weights=use_weights,
                seed=seed,
                other=other
            )
            exp.run_experiments()
    else:
        if args.train_loss is None:
            raise ValueError("You need to define train_loss!")
        elif args.comp_loss is None:
            raise ValueError(
                "For AGRA experiment you need to define both train_loss and comp_loss (or define none of them, and all "
                "combinations will be tried)."
            )
        else:
            exp = AGRAExperiments(
                input_data_path=args.path_to_data,
                dataset=args.dataset,
                output_data_path=args.output_path,
                extractor=args.extractor,
                model_name=None,
                metric=args.metric,
                metric_avg=None,
                crit=None,
                train_loss=args.train_loss,
                comp_loss=args.comp_loss,
                use_weights=use_weights,
                seed=seed,
                other=other
            )
            exp.run_experiments()
