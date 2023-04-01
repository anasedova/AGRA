import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.deep_agra.utils import define_model_name
from experiments.utils import get_cifar_data, get_mv_train_labels, load_train_labels_from_file, define_eval_metric
from src.DeepAGRA.bert_model_with_DeepAGRA import BertClassifierModelWithAGRA
from src.DeepAGRA.resnet_model_with_DeepAGRA import ResNetClassifierModelWithDeepAGRA
from src.utils import set_seed, compute_weights
from wrench.dataset import load_dataset

lr = 1e-3
batch_size = 16
agra_threshold = 0

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_train_data(dataset_path, dataset, train_labels_path, model_name, num_valid_samples: int = None):

    if dataset in ['youtube', 'sms', 'trec', 'yoruba', 'hausa']:

        # load wrench dataset
        train_dataset, valid_dataset, test_dataset = load_dataset(
            dataset_path, dataset, dataset_type='TextDataset', extract_fn="bert", model_name=model_name,
            extract_feature=False
        )
        # calculate train labels y_train with majority vote
        y_train = get_mv_train_labels(train_dataset)
        y_valid = valid_dataset.labels

    elif dataset == "cifar":
        train_dataset, valid_dataset, test_dataset, y_valid, y_test = get_cifar_data(
            os.path.join(dataset_path, dataset), num_valid_samples)

        # upload the labels from the file
        assert train_labels_path is not None
        train_labels_dict = load_train_labels_from_file(dataset_path, train_labels_path, dataset)

        train_dataset.samples = [(fn, train_labels_dict[fn]) for fn, _ in train_dataset.imgs]
        y_train = np.array([train_labels_dict[fn] for fn, _ in train_dataset.imgs])

    elif dataset == "chexpert":
        train_dataset, valid_dataset, test_dataset, y_train, y_valid = None, None, None, None, None
        pass

    else:
        raise ValueError(f"Dataset {dataset} is not yet supported")

    # todo: add gold labels
    # if args.gold_label_path is not None:
    #     with open(args.gold_label_path, 'r') as file:
    #         y_gold = np.load(file)

    return train_dataset, valid_dataset, test_dataset, y_train, y_valid


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
    parser.add_argument("--modification", type=str, default="last")
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--model", type=str, default='bert', choices=['bert', 'resnet'])  # or roberta-base
    parser.add_argument("--closs", type=str, default='CE', choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--save', type=bool, default=True)

    args = parser.parse_args()
    args.weights = True if args.weights == "True" else False

    # set the seed
    if args.seed is not None:
        set_seed(args.seed)

    # define the model to be trained
    model_name = define_model_name(args.model, args.dataset)

    # define the path from which the dataset will be loaded
    dataset_path = args.data_path if args.data_path else \
        os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')

    # define the name of the folder where the results will be stored
    results_folder = os.path.join(args.output_path, "results", 'single_run', 'deep_agra', args.dataset,
                                  f'model_{args.model}_weights_{args.weights}_comp_{args.closs}_other_{str(args.other)}')
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "result.txt"), "w")

    # define metric for the evaluation
    metric = define_eval_metric(args.dataset)

    # load model name, dataset and training labels (mv or separately provided)
    train_data, valid_data, test_data, train_labels, valid_labels = load_train_data(
        args.data_path, args.dataset, args.train_labels_path, model_name, args.num_valid_samples
    )

    # compute weights for comparison batch sampling
    agra_weights = compute_weights(train_labels) if args.weights else None

    # initialize model
    if args.model == 'bert':
        model = BertClassifierModelWithAGRA(
            model_name=model_name,
            agra_weights=agra_weights,
            comp_loss=args.closs,
            other=args.other,
            batch_size=batch_size,
            lr=lr,
            agra_threshold=agra_threshold
        )
    elif args.model == 'resnet':
        model = ResNetClassifierModelWithDeepAGRA(
            model_name=model_name,
            agra_weights=agra_weights,
            comp_loss=args.closs,
            other=args.other,
            batch_size=batch_size
        )
    else:
        raise ValueError('Model not implemented.')

    history = model.fit(
        dataset_train=train_data,
        y_train=train_labels,
        dataset_valid=valid_data,
        y_valid=valid_labels,

        modification=args.modification,
        metric=metric,
        device=device,
        evaluation_step=300,
        verbose=True
    )
    metric_value = model.test(DataLoader(test_data))
    print(f"Test {metric} value: {metric_value}")

    if args.save is True:
        output_file.write("\t".join(["model", args.model]) + "\n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["weights", args.weights]) + "\n")
        output_file.write("\t".join(["alternative_label", str(args.other)]) + "\n")
        output_file.write("\t".join(["lr", str(lr)]) + "\n")
        output_file.write("\t".join(["batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["agra_threshold", str(agra_threshold)]) + "\n")
        output_file.write("\t".join([metric, str(metric_value)]) + "\n")
