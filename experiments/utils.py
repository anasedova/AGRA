import copy
import json
import os
import pathlib
from typing import Callable, Optional, Union

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from wrench.evaluation import METRIC
from wrench.labelmodel.majority_voting import MajorityVoting
from snorkel.utils import probs_to_preds


def define_eval_metric(dataset):
    """
    Load the evaluation metric associated with the dataset.
    """
    if dataset in ['youtube', 'trec', 'cifar']:
        return 'acc'
    elif dataset == 'sms':
        return 'f1_binary'
    elif dataset in ['yoruba', 'hausa']:
        return 'f1_macro'
    elif dataset == 'chexpert':
        return 'macro_auc'
    else:
        raise ValueError('Dataset not available.')


def get_cifar_data(dataset_path, num_valid_samples, transformation="standard"):
    """
    Transform CIFAR data.
    """
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    valid_path = os.path.join(dataset_path, 'valid')

    if transformation == "cleanlab":
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
    elif transformation == "standard":
        #  source: https://pytorch.org/vision/0.8/models.html
        transform_train = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unsupported transformation {transformation}")

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
    """
    Retrieve majority vote labels.
    """
    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=data)
    soft_labels_train = label_model.predict_proba(data)
    return probs_to_preds(soft_labels_train)


def load_train_labels_from_file(dataset_path, train_labels_path, dataset):
    """
    Load noisy CIFAR-10 labels
    """
    assert train_labels_path is not None
    with open(train_labels_path, 'r') as rf:
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


def define_metric(metric: Optional[Union[str, Callable]]) -> None:
    # initialize metric
    return metric if isinstance(metric, Callable) else METRIC[metric]

def load_parameter_setting_for_youtube(loss: str, use_weights: str):
    """
    Loads the hyperparameter setting for each youtube combination
    :return: chosen batch size, learning rate and weight decay
    """

    if loss == 'CE' and use_weights == 'False':
        lr = 0.01
        batch_size = 32
        decay = 0.00001

    elif loss == 'F1' and use_weights == 'False':
        lr = 0.01
        batch_size = 32
        decay = 0.001

    elif loss == 'CE' and use_weights == 'True':
        lr = 0.01
        batch_size = 32
        decay = 0.00001

    elif loss == 'F1' and use_weights == 'True':
        lr = 0.01
        batch_size = 32
        decay = 0.001

    else:
        raise ValueError('Invalid loss and weights combination.')

    return batch_size, lr, decay

def load_parameter_setting(dataset: str):
    """
    Load chosen hyperparameters reported in paper.
    """

    if dataset == 'youtube':
        lr = 0.01
        batch_size = 32
        weight_decay = 0.001
        closs = 'F1'
        weights = 'False'
        other = None
        agra_threshold = 0
        num_epochs = 10
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'sms':
        lr = 0.001
        batch_size = 128
        weight_decay = 0.001
        closs = 'CE'
        weights = 'True'
        other = None
        agra_threshold = 0
        num_epochs = 10
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'trec':
        # gives significantly worse performance than in paper: CHECK!
        lr = 0.1
        batch_size = 32
        weight_decay = 0.0001
        closs = 'F1'
        weights = 'True'
        other = None
        agra_threshold = 0
        num_epochs = 10
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'yoruba':
        lr = 0.1
        batch_size = 512
        weight_decay = 0.0001
        closs = 'F1'
        weights = 'True'
        other = None
        agra_threshold = 0
        num_epochs = 10
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'hausa':
        lr = 0.1
        batch_size = 512
        weight_decay = 0.0001
        closs = 'F1'
        weights = 'True'
        other = None
        agra_threshold = 0
        num_epochs = 10
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'chexpert':
        lr = 0.001
        batch_size = 128
        weight_decay = 0.001
        closs = 'F1'
        weights = 'False'
        other = None
        agra_threshold = 0
        num_epochs = 5
        n_exp = 5
        encoding = "tfidf"

    elif dataset == 'cifar':
        pass

    else:
        raise ValueError('Invalid dataset.')

    return lr, batch_size, weight_decay, closs, weights, other, agra_threshold, num_epochs, n_exp, encoding
