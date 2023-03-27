import copy
import json
import os
import pathlib

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from wrench.labelmodel.majority_voting import MajorityVoting
from snorkel.utils import probs_to_preds


def define_data_encoding(args) -> str:
    # define the necessary encoding if not provided
    if args.encoding:
        print(f"The input data will be encoded with {args.encoding}")
        return args.encoding
    else:
        if args.dataset in ['youtube', 'sms', 'trec', 'yoruba', 'hausa']:
            return "tfidf"
        elif args.dataset in ["cifar", "chexpert"]:
            return "resnet50"
        else:
            raise ValueError(f"Unknown dataset {args.dataset}, please specify how you want to encode the data")


def define_eval_metric(dataset):
    if dataset in ['youtube', 'trec', 'cifar']:
        return 'acc'
    elif dataset == 'sms':
        return 'f1_binary'
    elif dataset in ['yoruba', 'hausa']:
        return 'f1_macro'
    else:
        raise ValueError('Dataset not available.')


def get_cifar_data(dataset_path, num_valid_samples):
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    valid_path = os.path.join(dataset_path, 'valid')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.ImageFolder(train_path, transform_train)
    test_data = datasets.ImageFolder(test_path, transform_test)

    if pathlib.Path(valid_path).exists():
        valid_data = datasets.ImageFolder(valid_path, transform_test)
        y_valid = valid_data.targets
        y_test = test_data.targets
    else:
        # validation set will be sampled from the test set
        valid_data = copy.deepcopy(test_data)
        valid_idx = np.random.choice(len(test_data), num_valid_samples, replace=False)
        test_idx = [idx for idx in list(range(len(test_data))) if idx not in valid_idx]

        valid_data = torch.utils.data.Subset(valid_data, valid_idx)
        test_data = torch.utils.data.Subset(test_data, test_idx)

        y_valid = np.array(test_data.dataset.targets)[np.ndarray.tolist(valid_idx)]
        y_test = np.array(test_data.dataset.targets)[test_idx]
    return train_data, valid_data, test_data, y_valid, y_test


def get_mv_train_labels(data):
    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=data)
    soft_labels_train = label_model.predict_proba(data)
    return probs_to_preds(soft_labels_train)


def load_train_labels_from_file(dataset_path, train_labels_path, dataset):
    assert train_labels_path is not None
    with open(os.path.join(dataset_path, dataset, train_labels_path), 'r') as rf:
        train_labels = json.load(rf)
    if dataset == "cifar":
        # additional postprocessing due to specificity of the cifar data
        updated_train_labels_dict = {}
        for key, value in train_labels.items():
            updated_train_labels_dict[
                os.path.join(dataset_path, dataset, 'train') + '/' + key.split('/')[-2] + '/' + key.split('/')[-1]
                ] = value
        train_labels = updated_train_labels_dict
    return train_labels
