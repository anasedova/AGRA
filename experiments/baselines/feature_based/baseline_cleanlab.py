import argparse
import os
import sys
import random
from tqdm import tqdm

import numpy as np
from cleanlab.classification import CleanLearning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score

from experiments.agra.run_agra import load_train_data_for_agra

num_exp = 5
num_grid_ext = 3

num_folds_values = [2, 5, 7, 9, 12]
learning_rates = np.logspace(-5, -1, num=5, base=10)
regularizations = np.logspace(-5, -3, num=3, base=10)
batch_sizes = [32, 128, 512]


def set_seed(seed: int) -> None:
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    random.seed(seed)
    np.random.seed(seed)


def run_cleanlab_baseline(
        path_to_data, dataset, train_labels_path, num_valid_samples: int, score: str, finetuning: bool = True,
        finetuning_epochs: int = 1000, finetuning_batch_size: int = 32, output_path: str = "./", seed: int = None
):
    if seed:
        set_seed(int(seed))

    output_file = os.path.join(output_path, f"{dataset}_out.txt")

    train_dataset, valid_dataset, test_dataset, y_train = load_train_data_for_agra(
        dataset, path_to_data, train_labels_path, num_valid_samples, finetuning=finetuning,
        finetuning_epochs=finetuning_epochs, finetuning_batch_size=finetuning_batch_size, metric=score
    )

    best_metric = 0
    best_setting = None

    for num_folds in num_folds_values:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for regularization in regularizations:
                    with open(output_file, "a+") as file:
                        file.write(
                            f"Setting: num_folds = {num_folds}, lr = {learning_rate}, batch_size = {batch_size} \n")
                        print(
                            f"Setting: num_folds = {num_folds}, lr = {learning_rate}, l2 = {regularization}, "
                            f"batch_size = {batch_size} \n")

                        metrics = []

                        for exp in range(num_grid_ext):
                            seed = np.random.randint(0, 1000)
                            set_seed(seed)

                            clf = MLPClassifier(hidden_layer_sizes=(), alpha=regularization,
                                                learning_rate_init=learning_rate, batch_size=batch_size,
                                                max_iter=10000)  # any classifier implementing the sklearn API
                            cl = CleanLearning(clf=clf, seed=seed, cv_n_folds=num_folds)
                            _ = cl.fit(train_dataset.features, np.array(y_train).astype(int))

                            cl_valid_preds = cl.predict(valid_dataset.features)

                            if score == "acc":
                                metric = accuracy_score(np.array(valid_dataset.labels), cl_valid_preds)
                            elif score == "f1_binary":
                                metric = f1_score(np.array(valid_dataset.labels), cl_valid_preds, average='binary',
                                                  pos_label=1)
                            elif score == "f1_macro":
                                metric = f1_score(np.array(valid_dataset.labels), cl_valid_preds, average='macro',
                                                  pos_label=1)
                            else:
                                raise ValueError("Invalid metric!")

                            with open(output_file, "a+") as file:
                                file.write(f"Exp: {exp}, {score}: {metric} \t")
                            print(f"Exp: {exp}, {score}: {metric} \t")

                            metrics.append(metric)
                        mean_metric = np.mean(metrics)

                        with open(output_file, "a+") as file:
                            file.write(f"\n Mean {score}: {mean_metric} \n")
                        print(f"\n Mean {score}: {mean_metric} \n")

                        if mean_metric > best_metric:
                            best_metric = mean_metric
                            best_setting = num_folds
                            best_lr = learning_rate
                            best_batch = batch_size
                            best_reg = regularization

    with open(output_file, "a+") as file:
        file.write(f"Selected setting is: cv_n_folds={best_setting} \n")
    print(f"Selected setting is: cv_n_folds={best_setting} \n")

    all_scores = []
    num_exp = 5

    for i in tqdm(range(num_exp)):

        seed = np.random.randint(0, 1000)
        set_seed(seed)

        clf = MLPClassifier(hidden_layer_sizes=(), alpha=best_reg, learning_rate_init=best_lr, batch_size=best_batch,
                            max_iter=10000)  # any classifier implementing the sklearn API
        cl = CleanLearning(clf=clf, seed=seed, cv_n_folds=best_setting)
        _ = cl.fit(train_dataset.features, np.array(y_train).astype(int))

        cl_test_preds = cl.predict(test_dataset.features)

        if score == "acc":
            metric = accuracy_score(np.array(test_dataset.labels), cl_test_preds)
        elif score == "f1_binary":
            metric = f1_score(np.array(test_dataset.labels), cl_test_preds, average='binary', pos_label=1)
        elif score == "f1_macro":
            metric = f1_score(np.array(test_dataset.labels), cl_test_preds, average='macro')
        else:
            raise ValueError("Invalid metric!")

        with open(output_file, "a+") as file:
            file.write(f"Experiment {i + 1}/{num_exp}: {score} is: {metric}\n")

        print(f"Experiment {i + 1}/{num_exp}: {score} is: {metric}")
        all_scores.append(metric)

    with open(output_file, "a+") as file:
        file.write(f"All scores: {all_scores}\n")
        file.write(f"Mean score is: {np.mean(np.array(all_scores))}, std: {np.std(np.array(all_scores))}")

    print(f"All scores: {all_scores}")
    print(f"Mean score is: {np.mean(np.array(all_scores))}, std: {np.std(np.array(all_scores))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--data_path", type=str, help="", default=None)
    # parser.add_argument("--dataset", help="", default='yoruba')
    parser.add_argument("--train_labels_path", type=str, help="", default=None)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_valid_samples", help="", default=5000)
    parser.add_argument("--finetuning", type=bool, default=True)
    parser.add_argument("--finetuning_epochs", type=int, default=100)
    parser.add_argument("--finetuning_batch_size", type=int, default=32)
    parser.add_argument("--seed", help="", default=0)
    args = parser.parse_args()

    for dataset in ["youtube", "sms", "spouse", "trec", "yoruba", "hausa"]:
        if args.data_path is None:
            path_to_data = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')
        else:
            path_to_data = args.data_path

        # define the name of the folder where the results will be stored
        results_folder = os.path.join(args.output_path, "results", 'grid_search', 'cleanlab')
        os.makedirs(results_folder, exist_ok=True)

        metrics = {
            "youtube": "acc",
            "sms": "f1_binary",
            "spouse": "f1_binary",
            "trec": "acc",
            "yoruba": "f1_macro",
            "hausa": "f1_macro",
            "cifar": "acc"
        }

        run_cleanlab_baseline(
            path_to_data=path_to_data,
            dataset=dataset,
            train_labels_path=args.train_labels_path,
            num_valid_samples=args.num_valid_samples,
            score=metrics[dataset],
            finetuning=args.finetuning,
            finetuning_epochs=args.finetuning_epochs,
            finetuning_batch_size=args.finetuning_batch_size,
            output_path=results_folder,
            seed=args.seed
        )

"""
--data_path
/Users/asedova/PycharmProjects/01_knodle/data
%
/Users/asedova/PycharmProjects/aaai_agra/data/
--dataset
cifar
%--train_labels_path
%/Users/asedova/PycharmProjects/aaai_agra/data/cifar/cifar10_train_noisy_labels__6_2.json
--output_path
/Users/asedova/PycharmProjects/aaai_agra/
--seed
12345
%
--finetuning
%
True
%
--finetuning_epochs
%
100
%
--finetuning_batch_size
%
64
"""
