import argparse
import logging
import os

import numpy as np
import torch
from torch import Tensor

from experiments.agra.utils import AGRAImageDataSet, load_image_dataset, define_data_encoding_agra
from experiments.baselines.feature_based.logreg_model_baseline import LogRegBaseline
from experiments.utils import define_eval_metric, get_mv_train_labels, get_cifar_data, \
    load_train_labels_from_file
from src.utils import set_seed
from wrench.dataset import load_dataset, BaseDataset

logger = logging.getLogger(__name__)

lr = 1e-1
batch_size = 32
agra_threshold = 0

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_train_data_for_agra(
        dataset, data_path, train_labels_path: str = None, num_valid_samples: int = None, enc_model: str = "resnet50",
        finetuning: bool = False, finetuning_epochs: int = 2, metric: str = "acc"
):
    if dataset in ['youtube', 'sms', 'trec', 'yoruba', 'hausa']:
        # load wrench dataset
        train_data, test_data, valid_data = load_dataset(
            data_path, dataset, dataset_type='TextDataset', extract_fn="tfidf", extract_feature=True
        )
        # calculate train labels y_train with majority vote
        y_train = get_mv_train_labels(train_data)
        train_data.labels = y_train

    elif dataset in ['cifar', 'chexpert']:

        # load datasets
        train_data, test_data, valid_data, y_valid, y_test = get_cifar_data(
            os.path.join(data_path, dataset), num_valid_samples)

        # upload the labels from the file
        train_labels_dict = load_train_labels_from_file(dataset_path, train_labels_path, dataset)
        y_train = np.array(list(train_labels_dict.values()))

        # calculate num_classes
        num_classes = max(int(max(y_train)), int(max(y_valid)), int(max(y_test))) + 1

        # load Cifar and CheXpert datasets and get encodings with resnet-50
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_image_dataset(
            data_path, dataset, train_data, test_data, valid_data, enc_model, num_classes=num_classes,
            finetuning=finetuning, finetuning_epochs=finetuning_epochs, metric=metric, batch_size=FT_batch_size
        )

        # transform the data into wrench-compatible datasets
        train_data = AGRAImageDataSet(Tensor(train_features), Tensor(train_labels))
        valid_data = AGRAImageDataSet(Tensor(valid_features), Tensor(valid_labels))
        test_data = AGRAImageDataSet(Tensor(test_features), Tensor(test_labels))

    else:
        raise ValueError(f"Dataset {dataset} is not yet supported.")

    return train_data, valid_data, test_data, y_train


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
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()

    print(args)
    print(f"FT batch size: {FT_batch_size}")

    # set the seed
    if args.seed is not None:
        set_seed(args.seed)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding_agra(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'single_run', 'baseline', args.dataset)
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "w")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load dataset and training labels (mv or separately provided)
    train_dataset, valid_dataset, test_dataset, train_labels = load_train_data_for_agra(
        args.dataset, args.data_path, args.train_labels_path, args.num_valid_samples, encoding,
        finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs, metric=metric)

    y_train = torch.Tensor(train_dataset.weak_labels)

    model = LogRegBaseline()
    model.fit(train_dataset, valid_dataset, metric=metric)

    if isinstance(test_dataset, BaseDataset):
        metric_value = model.test(test_dataset, metric)
    else:
        test_dataloader = model._init_valid_dataloader(test_dataset)
        metric_value = model.test(test_dataloader, y_true=test_dataset.labels, metric_fn=metric)
    print(metric_value)

    if args.save is True:
        output_file.write("Baseline with LogReg \n")
        output_file.write("\t".join(["model", "logreg"]) + "\n")
        output_file.write("\t".join(["lr", str(lr)]) + "\n")
        output_file.write("\t".join(["batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["finetuning", str(args.finetuning)]) + "\n")
        output_file.write("\t".join(["finetuning_epochs", str(args.finetuning_epochs)]) + "\n")
        output_file.write("\t".join([metric, str(metric_value)]) + "\n")
