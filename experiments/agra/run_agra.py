import argparse
import os

import numpy as np
import torch
from torch import Tensor

from experiments.agra.utils import AGRAImageDataSet, load_image_dataset, define_data_encoding_agra
from experiments.utils import define_eval_metric, get_mv_train_labels, get_cifar_data, \
    load_train_labels_from_file
from src.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from src.utils import set_seed, compute_weights
from wrench.dataset import load_dataset, BaseDataset

lr = 1e-2
batch_size = 32
agra_threshold = 0

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
    parser.add_argument("--closs", type=str, default='CE', choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()

    print(args)

    # set the seed
    if args.seed is not None:
        print(f"Seed {args.seed} is fixed.")
        set_seed(args.seed)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding_agra(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'single_run', 'agra', args.dataset)
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "a+")

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

    # compute weights for comparison batch sampling
    agra_weights = compute_weights(train_labels) if args.weights == 'True' else None

    # initialize the model
    model = LogRegModelWithAGRA(
        num_classes=num_classes,
        agra_weights=agra_weights,
        comp_loss=args.closs,
        other=args.other,
        lr=lr,
        batch_size=batch_size,
        agra_threshold=agra_threshold
    )

    history = model.fit(
        dataset_train=train_dataset,
        dataset_valid=valid_dataset,

        modification=args.modification,
        metric=metric,
        device=device,
        evaluation_step=100,
        verbose=True,

        patience=50            # 20 by default
    )
    if isinstance(test_dataset, BaseDataset):
        metric_value = model.test(test_dataset, metric)
    else:
        test_dataloader = model._init_valid_dataloader(test_dataset)
        metric_value = model.test(test_dataloader, y_true=test_dataset.labels, metric_fn=metric)
    print(metric_value)

    if args.save is True:
        output_file.write("AGRA with LogReg \n")
        output_file.write("\t".join(["model", "logreg"]) + "\n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["weights", args.weights]) + "\n")
        output_file.write("\t".join(["alternative_label", str(args.other)]) + "\n")
        output_file.write("\t".join(["lr", str(lr)]) + "\n")
        output_file.write("\t".join(["batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["FT_batch_size", str(args.finetuning_batch_size)]) + "\n")
        output_file.write("\t".join(["agra_threshold", str(agra_threshold)]) + "\n")
        output_file.write("\t".join(["finetuning", str(args.finetuning)]) + "\n")
        output_file.write("\t".join(["finetuning_epochs", str(args.finetuning_epochs)]) + "\n")
        output_file.write("\t".join([metric, str(metric_value)]) + "\n" + "\n")