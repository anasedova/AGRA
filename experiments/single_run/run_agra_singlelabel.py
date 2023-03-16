import argparse
import copy
import json
import os
import pathlib

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from snorkel.utils import probs_to_preds

from _scr.AGRA.logreg_model_with_AGRA import LogRegModelWithAGRA
from _scr.DeepAGRA.bert_model_with_AGRA import BertClassifierModelWithAGRA
from _scr.DeepAGRA.resnet_model_with_AGRA import ResNetClassifierModelWithAGRA
from _scr.utils import set_seed, compute_weights
from wrench.dataset import load_dataset
from wrench.labelmodel.majority_voting import MajorityVoting

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_labels_path", type=str, default=None)
    parser.add_argument("--gold_label_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='youtube',
                        choices=['spam', 'sms', 'trec', 'yoruba', 'hausa', 'cifar'])
    parser.add_argument("--num_valid_samples", type=int, default=5000, help="Size of the validation set to be sampled "
                                                                            "from the test set if no validation set is "
                                                                            "available")
    parser.add_argument("--other", type=int, default=None)
    parser.add_argument("--model", type=str, default='logreg', choices=['logreg', 'bert', 'resnet'])  # or roberta-base
    parser.add_argument("--closs", type=str, default='CE', choices=['CE', 'F1'])
    parser.add_argument("--weights", type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_path is None:
        dataset_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')
    else:
        dataset_path = args.dataset_path

    results_folder = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'results', 'agra', args.dataset, f'model_{args.model}_weights_{args.weights}_comp_{args.closs}_other_{str(args.other)}')
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "w")

    if args.dataset in ['youtube', 'trec', 'cifar']:
        metric = 'acc'
    elif args.dataset == 'sms':
        metric = 'f1_binary'
    elif args.dataset in ['yoruba', 'hausa']:
        metric = 'f1_macro'
    else:
        raise ValueError('Dataset not available.')

    lr = 1e-3
    batch_size = 8
    agra_threshold = 0

    y_valid = None

    # load training data
    if args.model == 'logreg':
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            args.dataset,
            dataset_type='TextDataset',
            extract_feature=True,
            extract_fn='tfidf'
        )
    elif args.model == 'bert':
        if args.dataset in ['yoruba', 'hausa']:
            model_name = 'bert-base-multilingual-cased'
        else:
            model_name = 'roberta-base'  # todo: try distilbert
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            args.dataset,
            dataset_type='TextDataset',
            extract_feature=False
        )

    elif args.model == 'resnet':
        model_name = 'resnet50'
        train_path = os.path.join(args.dataset_path, 'train')
        test_path = os.path.join(args.dataset_path, 'test')
        valid_path = os.path.join(args.dataset_path, 'valid')

        train_data = datasets.ImageFolder(
            train_path,
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        )

        test_data = datasets.ImageFolder(
            test_path,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        )

        if pathlib.Path(valid_path).exists():
            valid_data = datasets.ImageFolder(
                valid_path,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            )
        else:
            # validation set will be sampled from the test set
            valid_data = copy.deepcopy(test_data)
            valid_idx = np.random.choice(len(test_data), args.num_valid_samples, replace=False)
            test_idx = [idx for idx in list(range(len(test_data))) if idx not in valid_idx]

            valid_data = torch.utils.data.Subset(valid_data, valid_idx)
            test_data = torch.utils.data.Subset(test_data, test_idx)

            y_valid = np.array(test_data.dataset.targets)[np.ndarray.tolist(valid_idx)]

    else:
        raise ValueError('Model not implemented.')

    # calculate weak labels with majority vote
    if args.train_labels_path is None:
        # get the labels by majority voting for train data
        label_model = MajorityVoting()
        label_model.fit(dataset_train=train_data)
        soft_labels_train = label_model.predict_proba(train_data)
        y_train = probs_to_preds(soft_labels_train)

    # load noisy labels from the file
    else:
        with open(args.train_labels_path, 'r') as rf:
            train_labels_dict = json.load(rf)
        updated_train_labels_dict = {}

        for key, value in train_labels_dict.items():
            updated_train_labels_dict[train_path + '/' + key.split('/')[-2] + '/' + key.split('/')[-1]] = value
        train_labels_dict = updated_train_labels_dict

        train_data.samples = [(fn, train_labels_dict[fn]) for fn, _ in train_data.imgs]
        y_train = np.array([train_labels_dict[fn] for fn, _ in train_data.imgs])
        # if args.gold_label_path is not None:
        #     with open(args.gold_label_path, 'r') as file:
        #         y_gold = np.load(file)

    # compute weights for comparison batch sampling
    if args.weights == 'True':
        agra_weights = compute_weights(y_train)
    else:
        agra_weights = None

    # initialize model
    if args.model == 'logreg':
        model = LogRegModelWithAGRA(
            agra_weights=agra_weights,
            comp_loss=args.closs,
            other=args.other,
            lr=lr,
            batch_size=batch_size,
            agra_threshold=agra_threshold
        )
    elif args.model == 'bert':
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
        model = ResNetClassifierModelWithAGRA(
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
        y_train=y_train,
        dataset_valid=valid_data,
        y_valid=y_valid,
        metric=metric,
        device=device,
        evaluation_step=300,
        verbose=False
    )
    metric_value = model.test(test_data, metric)
    print(metric_value)

    if args.save is True:
        output_file.write("\t".join(["model", args.model]) + "\n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["weights", args.weights]) + "\n")
        output_file.write("\t".join(["alternative_label", str(args.other)]) + "\n")
        output_file.write("\t".join(["lr", str(lr)]) + "\n")
        output_file.write("\t".join(["batch_size", str(batch_size)]) + "\n")
        output_file.write("\t".join(["agra_threshold", str(agra_threshold)]) + "\n")
        output_file.write("\t".join([metric, str(metric)]) + "\n")