import argparse
import logging
import os

import joblib
import numpy as np
import statistics
from scipy import stats

from depr.src import train_grad_match_multilabel
from depr.src.multi_label.utils_multilabel import set_seed, load_data

logger = logging.getLogger(__name__)

epochs = 5
# number of final experiments to be performed with the best parameter combination
num_final_exp = 5

# parameter values
learning_rate = 0.001
batch_size = 128
weight_decay = 0.001
threshold = -0.7

train_loss = "F1"
comp_loss = "F1"


class AGRAExperiments:

    def __init__(
            self,
            input_data_path: str,
            output_data_path: str,
            metric: str,
            seed: int = None,
            save_preds: bool = False
    ):

        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.metric = metric
        self.save_preds = save_preds

        if seed:
            set_seed(int(seed))

        os.makedirs(self.output_data_path, exist_ok=True)

        # load data from csv files
        self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test = load_data(self.input_data_path)

        if self.save_preds:
            joblib.dump(self.y_test, f"./gold")

    def run_experiments(self):
        # run GM experiments
        param_combs = (learning_rate, batch_size, weight_decay, threshold)

        dev_performances, test_performances = self.final(param_combs)  # run final experiments

        # maybe remove/adapt (not sure how you need it)
        with open(os.path.join(self.output_data_path, f"{param_combs}_best"), "a+") as file:
            file.write(
                f"All dev performances: {dev_performances} \n All test performances: {test_performances} \n"
                f"Mean Macro-AUROC on dev set: {round(sum(dev_performances) / len(dev_performances), 5)}, "
                f"std: {round(statistics.stdev(dev_performances), 5)}, sem: {round(stats.sem(dev_performances), 5)} \n"
                f"Mean Macro-AUROC on test set: {round(sum(test_performances) / len(test_performances), 5)}, "
                f"std: {round(statistics.stdev(test_performances), 5)}, sem: {round(stats.sem(test_performances), 5)}\n"
            )

    def final(self, chosen_comb):

        dev_performances, test_performances = [], []

        learning_rate = chosen_comb[0]
        batch_size = chosen_comb[1]
        decay = chosen_comb[2]
        thresh = chosen_comb[3]

        for i in range(num_final_exp):

            seed = np.random.randint(0, 1000)
            set_seed(seed)

            if self.output_data_path is not None:
                storing_loc_models = os.path.join(
                    self.output_data_path, f"{chosen_comb}_best_{str(i + 1)}"
                )
            else:
                storing_loc_models = None

            val_auroc, test_auroc, ignored_0, ignored_1, kept_0, kept_1, dev_probs, test_probs = \
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
                    metric=self.metric,
                    weight_decay=decay,
                    loss_reduction="avg_per_class",  # for BCE loss
                    thresh=thresh,
                    storing_loc=storing_loc_models
                )
            if self.save_preds:
                joblib.dump(test_probs, f"./preds_{i}")

            dev_performances.append(val_auroc)
            test_performances.append(test_auroc)

        return dev_performances, test_performances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', default='', type=str)
    parser.add_argument('--output_data_path', default='', type=str)
    parser.add_argument('--metric', default='', type=str)
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    logger.info(args)

    exp = GMExperiments(
        input_data_path=args.path_to_data,
        output_data_path=args.output_data_path,
        metric=args.metric,
        seed=args.seed
    )
    exp.run_experiments()
