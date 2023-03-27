import argparse
import os

import numpy as np
import torch
from torch import Tensor

from experiments.agra.utils import AGRAImageDataSet, load_image_dataset
from experiments.utils import define_data_encoding, define_eval_metric, get_mv_train_labels, get_cifar_data, \
    load_train_labels_from_file
from src.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from src.utils import set_seed, compute_weights
from wrench.dataset import load_dataset, BaseDataset

lr = 1e-2
batch_size = 128
agra_threshold = 0

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_train_data_for_agra(
        dataset, data_path, train_labels_path: str = None, num_valid_samples: int = None, encoding: str = "resnet50",
        finetuning: bool =False, finetuning_epochs : int = 2
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
        # assert that the label file is also provided
        assert train_labels_path is not None

        # load datasets
        train_data, test_data, valid_data, y_valid, y_test = get_cifar_data(
            os.path.join(data_path, dataset), num_valid_samples)

        # load Cifar and CheXpert datasets and get encodings with resnet-50
        train_features, valid_features, test_features = load_image_dataset(
            data_path, dataset, train_data, test_data, valid_data, encoding, finetuning=finetuning,
            finetuning_epochs=finetuning_epochs
        )

        # upload the labels from the file
        assert train_labels_path is not None
        train_labels_dict = load_train_labels_from_file(dataset_path, train_labels_path, dataset)
        y_train = np.array(list(train_labels_dict.values()))

        # transform the data into wrench-compatible datasets
        train_data = AGRAImageDataSet(Tensor(train_features), Tensor(y_train))
        valid_data = AGRAImageDataSet(Tensor(valid_features), Tensor(y_valid))
        test_data = AGRAImageDataSet(Tensor(test_features), Tensor(y_test))

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
                        choices=['youtube', 'sms', 'trec', 'yoruba', 'hausa', 'cifar'])
    parser.add_argument("--num_valid_samples", type=int, default=5000,
                        help="Size of a valid set to be sampled from the test set if no valid set is available")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--finetuning_epochs", type=int, default=2)
    parser.add_argument("--modification", type=str, default="last")
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--closs", type=str, default='CE', choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # set the seed
    if args.seed is not None:
        set_seed(args.seed)

    # define the way how the dataset will be encoded
    encoding = define_data_encoding(args)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'single_run', 'agra', args.dataset,
                                  f'model_logreg_weights_{args.weights}_comp_{args.closs}_other_{str(args.other)}')
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "w")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load dataset and training labels (mv or separately provided)
    train_dataset, valid_dataset, test_dataset, train_labels = load_train_data_for_agra(
        args.dataset, args.data_path, args.train_labels_path, args.num_valid_samples, encoding,
        finetuning=args.finetuning, finetuning_epochs=args.finetuning_epochs)

    # compute weights for comparison batch sampling
    agra_weights = compute_weights(train_labels) if args.weights == 'True' else None

    # initialize the model
    model = LogRegModelWithAGRA(
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
        output_file.write("\t".join(["agra_threshold", str(agra_threshold)]) + "\n")
        output_file.write("\t".join([metric, str(metric)]) + "\n")

