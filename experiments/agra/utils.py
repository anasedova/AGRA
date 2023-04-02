import copy
import os
import pickle
from typing import List, Tuple, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.utils import get_mv_train_labels, load_train_labels_from_file, get_cifar_data
from wrench.dataset import BaseDataset, load_dataset

from wrench.evaluation import METRIC

# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AGRAImageDataSet(TensorDataset, BaseDataset):
    def __init__(self, *tensors: Tensor, **kwargs):
        super(TensorDataset, self).__init__(**kwargs)
        self.tensors = tensors

        self.ids = []
        self.features = []
        self.labels = []            # todo: add gold labels here
        self.weak_labels = []
        self.examples = []

        if len(self.tensors) > 0:
            self.add_values()

    def create_subset(self, idx: List[int]):
        dataset = self.__class__()
        for i in idx:
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])

        if self.features is not None:
            dataset.features = self.features[idx]

        return dataset

    def add_values(self):
        self.ids = list(range(len(self.tensors[0])))
        self.features = self.tensors[0]
        self.labels = self.tensors[1]
        self.weak_labels = self.tensors[1]  # same as self.labels
        self.examples = self.tensors[0]  # same as self.features


    def extract_feature_(self, **kwargs):
        pass

def define_data_encoding_agra(args) -> str:
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


def get_resnet_embedding(train_loader: DataLoader, f_model: Any = None) -> np.ndarray:
    """
    Obtaining ResNet embeddings
    https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/
    https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    """
    out_embeddings = []

    def copy_embedding(m, i, o):
        o = o[:, :, 0, 0].cpu().detach().numpy()
        out_embeddings.append(o)

    layer = f_model._modules.get('avgpool')
    layer.register_forward_hook(copy_embedding)

    f_model.eval()            # no dropout layers are active
    all_labels = []
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        all_labels.append(y.cpu().detach().numpy())
        f_model(X)
    embeddings = [item for sublist in out_embeddings for item in sublist]
    embeddings = np.stack(embeddings, axis=0)
    all_labels = np.concatenate(all_labels)

    # print(len(list_embeddings))           # 50000
    # print(np.array(list_embeddings[0]).shape)         # 2048
    return embeddings, all_labels


def finetune_resnet(
        f_model, train_loader, valid_loader, epochs=2, metric="acc", lr=1e-1, momentum=0.9, weight_decay=0.0005,
        lr_decay=0.1
):
    """ Fine-tune the pretrained model (e.g. ResNet) before extracting the embeddings """

    assert metric in METRIC.keys()
    metric_fn = METRIC[metric]

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # optimizer = AdamW(f_model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(f_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)    #  CORES
    optimizer = AdamW(f_model.parameters(), lr=1e-3, weight_decay=0.0)            # the setting from 29.03
    f_model.train()
    optimizer.zero_grad()

    best_metric, best_epoch = 0, 0
    for n_epoch, epoch in enumerate(range(epochs)):
        print(f"Epoch {n_epoch + 1}/{epochs}...")
        for batch, labels in tqdm(train_loader):
            optimizer.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            output = f_model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        f_model.eval()
        # evaluate the model performance
        predictions_list, label_list = [], []
        with torch.no_grad():
            for val_batch, val_labels in valid_loader:
                val_batch, val_labels = val_batch.to(device), val_labels.to(device)
                output = f_model(val_batch)
                predictions = output.detach().cpu().numpy()
                predictions_list.append(predictions)
                label_list.append(val_labels.detach().cpu().numpy())
        predictions = np.concatenate(predictions_list)
        gold_labels = np.squeeze(np.hstack(label_list))
        valid_metric = metric_fn(gold_labels, predictions)
        print(f"Current validation {metric} is {valid_metric}")
        f_model.train()

        if valid_metric > best_metric:
            best_metric = valid_metric
            best_model = copy.deepcopy(f_model.state_dict())
            best_epoch = n_epoch

    # f_model.load_state_dict(best_model)
    # print(f"The best metric {best_metric} was achieved at epoch {best_epoch}; "
    #       f"this model was used to calculate the embeddings.")
    print(f"The model was fine-tuned for {epochs} epochs. ")

    return f_model


def get_resnet_embedding_ver_2(train_loader):
    """
    https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008/2
    Another version how to obtain the ResNet embeddings. Result is equal to get_resnet_embedding output
    """

    # set_seed(12345)
    resnet_model = models.resnet50(pretrained=True)
    modules = list(resnet_model.children())[:-1]
    resnet_model = nn.Sequential(*modules)
    for p in resnet_model.parameters():
        p.requires_grad = False

    # set_seed(12345)
    # resnet_model.eval()
    all_features = []
    for X, y in train_loader:
        features = resnet_model(X).data  # get the output from the last hidden layer of the pretrained resnet -> tensor
        all_features.append(torch.squeeze(features).numpy())
    embeddings = [item for sublist in all_features for item in sublist]
    embeddings = np.concatenate(embeddings, axis=0)

    # print(len(embeddings))           # 50000
    # print(np.array(embeddings[0]).shape)         # 2048
    return embeddings


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
        # train_data.labels = train_labels

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


def load_image_dataset(
        data_path, dataset, train_data, test_data, valid_data, encoding: str = "resnet50", num_classes: int = None,
        batch_size: int = 32, finetuning: bool = False, finetuning_epochs: int = 2, metric: str = "acc"
):
    """ Load the train, valid and test sets for image dataset (currently: Cifar, CheXpert) """

    set = f"_finetuned_epoch{finetuning_epochs}_batchsize{batch_size}_old_setting" if finetuning else ""
    path_to_cache = os.path.join(data_path, dataset, f"encoded_{encoding}{set}")

    if os.path.exists(path_to_cache):
        train_embeddings = joblib.load(os.path.join(path_to_cache, f"train_embeddings_{encoding}.data"))
        valid_embeddings = joblib.load(os.path.join(path_to_cache, f"valid_embeddings_{encoding}.data"))
        test_embeddings = joblib.load(os.path.join(path_to_cache, f"test_embeddings_{encoding}.data"))

        train_labels = joblib.load(os.path.join(path_to_cache, f"train_labels.data"))
        valid_labels = joblib.load(os.path.join(path_to_cache, f"valid_labels.data"))
        test_labels = joblib.load(os.path.join(path_to_cache, f"test_labels.data"))

        if type(train_labels) is list:
            train_labels = np.concatenate(train_labels)
            assert train_labels.shape[0] == train_embeddings.shape[0]

        if type(valid_labels) is list:
            valid_labels = np.concatenate(valid_labels)
            assert valid_labels.shape[0] == valid_embeddings.shape[0]

        if type(test_labels) is list:
            test_labels = np.concatenate(test_labels)
            assert test_labels.shape[0] == test_embeddings.shape[0]

        print(f"Embeddings are loaded from {path_to_cache}")
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

        if encoding == "resnet50":
            f_model = models.resnet50(pretrained=True).to(device)
        else:
            raise ValueError(f"The encoding {encoding} is not yet supported.")

        if finetuning:
            f_model.fc = nn.Linear(f_model.fc.in_features, num_classes)
            f_model = f_model.to(device)
            print(f"FINE-TUNING \t {encoding} model \t epochs: {finetuning_epochs}")
            f_model = finetune_resnet(f_model, train_loader, valid_loader, epochs=finetuning_epochs, metric=metric)

        train_embeddings, train_labels = get_resnet_embedding(train_loader, f_model)
        valid_embeddings, valid_labels = get_resnet_embedding(valid_loader, f_model)
        test_embeddings, test_labels = get_resnet_embedding(test_loader, f_model)

        os.makedirs(path_to_cache, exist_ok=True)
        with open(os.path.join(path_to_cache, f"train_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(train_embeddings, file)
        with open(os.path.join(path_to_cache, f"train_labels.data"), 'wb') as file:
            pickle.dump(train_labels, file)

        with open(os.path.join(path_to_cache, f"valid_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(valid_embeddings, file)
        with open(os.path.join(path_to_cache, f"valid_labels.data"), 'wb') as file:
            pickle.dump(valid_labels, file)

        with open(os.path.join(path_to_cache, f"test_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(test_embeddings, file)
        with open(os.path.join(path_to_cache, f"test_labels.data"), 'wb') as file:
            pickle.dump(test_labels, file)

        print(f"New embeddings are calculated and saved to {path_to_cache}")

    return train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels

