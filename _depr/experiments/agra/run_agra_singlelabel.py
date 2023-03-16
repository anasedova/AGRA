import logging
import os
import statistics

import numpy as np
from scipy import stats

from depr.src.single_label.gradient_matching_single_label import train_grad_match_with_gold
from depr.src import set_seed
from depr.src.single_label.wrench_dataload import prepare_data

logger = logging.getLogger(__name__)

### define the input parameters
dataset = "sms"
metric = "F1"
input_data_path = "/path/to/datasets"
storing_loc = "./results"
os.makedirs(storing_loc, exist_ok=True)
extractor = "tfidf"
model_name = None
epochs = 10  # number of epochs

use_weights = True
other = None
train_loss = "CE"
comp_loss = "CE"
learning_rate = 0.001
batch_size = 32
decay = 0.01
thresh = -0.4

num_final_exp = 5

train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, gold_labels, mislabeled, \
    weights, noise_rate = prepare_data(dataset, "TextDataset", input_data_path, extractor)

sample_weights = weights if use_weights else None

# which averaging will be used for AGRA update (when optimizing towards the F1 score)
metric_avg = "binary" if max(gold_labels) == 1 else "macro"
# which averaging will be used for evaluation score calculation
crit = "binary" if max(gold_labels) == 1 else "macro"

dev_performances = []
test_performances = []

for i in range(num_final_exp):
    seed = np.random.randint(0, 1000)
    set_seed(seed)
    # train gradient matching with specified parameters
    acc, f1, precision, recall, acc_test, f1_test, precision_test, recall_test, ignored, ignored_mislabeled, \
    ignored_correctly_labeled, mislabeled_ignored, correctly_corrected, corrected, test_predictions = \
        train_grad_match_with_gold(
            train_sentences_ds,
            train_labels_ds,
            dev_sentences,
            dev_labels,
            test_sentences,
            test_labels,
            gold_labels=gold_labels,
            mislabeled=mislabeled,
            other_class=other,
            seed=seed,
            num_epochs=epochs,
            batch_size=batch_size,
            weight_decay=decay,
            comp_batch_size=batch_size,
            learning_rate=learning_rate,
            training_loss=train_loss,
            comparison_loss=comp_loss,
            sample_weights=sample_weights,
            metric=metric,
            metric_avg=metric_avg,
            crit=crit,
            threshold=thresh,
            print_plots=True,
            smooth=30
        )

    if metric == "acc":
        dev_performances.append(acc)
        test_performances.append(acc_test)

    if metric == "F1":
        dev_performances.append(f1)
        test_performances.append(f1_test)

    print(
        {"acc": acc, "f1": f1, "precision": precision, "recall": recall, "acc_test": acc_test,
         "f1_test": f1_test, "precision_test": precision_test, "recall_test": recall_test, "ignored": ignored,
         "ignored_mislabeled": ignored_mislabeled, "ignored_correctly_labeled": ignored_correctly_labeled,
         "mislabeled_ignored": mislabeled_ignored, "correctly_corrected": correctly_corrected,
         "corrected": corrected}
    )

    logger.info("==================================================")


logger.info(f"All dev {metric}: {dev_performances} \n All test performances: {test_performances}")
logger.info(
    f"Mean {metric} on dev set: {round(sum(dev_performances) / len(dev_performances), 5)}, "
    f"std: {round(statistics.stdev(dev_performances), 5)}, sem: {round(stats.sem(dev_performances), 5)}"
)
logger.info(
    f"Mean {metric} on test set: {round(sum(test_performances) / len(test_performances), 5)}, "
    f"std: {round(statistics.stdev(test_performances), 5)}, sem: {round(stats.sem(test_performances), 5)}"
)

