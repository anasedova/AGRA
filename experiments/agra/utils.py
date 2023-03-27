import json
import os
import pickle
from typing import List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


# set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AGRAImageDataSet(TensorDataset):
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


def get_resnet_embedding(
        train_loader: DataLoader, encoding: str = "resnet50", finetuning: bool = False, epochs: int = 2
) -> np.ndarray:
    """
    https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/
    https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    # todo: is resnet_model.eval() really required here? = no dropout layers are active
    """
    out_embeddings = []

    def copy_embedding(m, i, o):
        o = o[:, :, 0, 0].detach().numpy()
        out_embeddings.append(o)

    if encoding == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"The encoding {encoding} is not yet supported.")

    if finetuning:
        model = train_resnet(model, train_loader, epochs)

    layer = model._modules.get('avgpool')
    layer.register_forward_hook(copy_embedding)

    model.eval()
    for X, y in train_loader:
        model(X)
    embeddings = [item for sublist in out_embeddings for item in sublist]
    embeddings = np.stack(embeddings, axis=0)

    # print(len(list_embeddings))           # 50000
    # print(np.array(list_embeddings[0]).shape)         # 2048
    return embeddings


def train_resnet(model, loader, epochs=2, lr=1e-3, l2=0.0):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=l2)
    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


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


def load_image_dataset(
        data_path, dataset, train_data, test_data, valid_data, encoding: str = "resnet50", batch_size: int = 128,
        finetuning: bool = False, finetuning_epochs: int = 2
) -> Tuple[List[List], List[List], List[List]]:
    """ Load the train, valid and test sets for image dataset (currently: Cifar, CheXpert) """

    set = "_finetuned" if finetuning else ""
    if os.path.exists(os.path.join(data_path, dataset, f"encoded_{encoding}{set}")):
        train_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"train_embeddings_{encoding}.data"))
        valid_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"valid_embeddings_{encoding}.data"))
        test_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"test_embeddings_{encoding}.data"))
        print(f"Embeddings are loaded from {os.path.join(data_path, dataset, f'encoded_{encoding}')}")
    else:

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

        train_embeddings = get_resnet_embedding(train_loader, encoding, finetuning=finetuning, epochs=finetuning_epochs)
        valid_embeddings = get_resnet_embedding(valid_loader, encoding, finetuning=finetuning, epochs=finetuning_epochs)
        test_embeddings = get_resnet_embedding(test_loader, encoding, finetuning=finetuning, epochs=finetuning_epochs)

        with open(os.path.join(
                data_path, dataset, f"encoded_{encoding}", f"train_embeddings_{encoding}{set}.data"), 'wb'
        ) as file:
            pickle.dump(train_embeddings, file)

        with open(os.path.join(
                data_path, dataset, f"encoded_{encoding}", f"valid_embeddings_{encoding}{set}.data"), 'wb'
        ) as file:
            pickle.dump(valid_embeddings, file)

        with open(os.path.join(
                data_path, dataset, f"encoded_{encoding}", f"test_embeddings_{encoding}{set}.data"), 'wb'
        ) as file:
            pickle.dump(test_embeddings, file)

    return train_embeddings, valid_embeddings, test_embeddings

