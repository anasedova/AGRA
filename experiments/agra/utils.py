import json
import os
import pickle
from typing import List, Tuple, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        f_model(X)
    embeddings = [item for sublist in out_embeddings for item in sublist]
    embeddings = np.stack(embeddings, axis=0)

    # print(len(list_embeddings))           # 50000
    # print(np.array(list_embeddings[0]).shape)         # 2048
    return embeddings


def finetune_resnet(f_model, loader, epochs=2, lr=1e-3, l2=0.0):
    """ Fine-tune the pretrained model (e.g. ResNet) before extracting the embeddings """
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(f_model.parameters(), lr=lr, weight_decay=l2)
    f_model.train()
    optimizer.zero_grad()
    for n_epoch, epoch in enumerate(range(epochs)):
        print(f"Epoch {n_epoch}...")
        for batch, labels in tqdm(loader):
            batch, labels = batch.to(device), labels.to(device)
            output = f_model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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


def load_image_dataset(
        data_path, dataset, train_data, test_data, valid_data, encoding: str = "resnet50", num_classes: int = None,
        batch_size: int = 128, finetuning: bool = False, finetuning_epochs: int = 2
) -> Tuple[List[List], List[List], List[List]]:
    """ Load the train, valid and test sets for image dataset (currently: Cifar, CheXpert) """

    set = f"_finetuned_epoch{finetuning_epochs}" if finetuning else ""
    path_to_cache = os.path.join(data_path, dataset, f"encoded_{encoding}{set}")
    if os.path.exists(path_to_cache):
        train_embeddings = joblib.load(os.path.join(path_to_cache, f"train_embeddings_{encoding}.data"))
        valid_embeddings = joblib.load(os.path.join(path_to_cache, f"valid_embeddings_{encoding}.data"))
        test_embeddings = joblib.load(os.path.join(path_to_cache, f"test_embeddings_{encoding}.data"))
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
            f_model = finetune_resnet(f_model, train_loader, finetuning_epochs)

        train_embeddings = get_resnet_embedding(train_loader, f_model)
        valid_embeddings = get_resnet_embedding(valid_loader, f_model)
        test_embeddings = get_resnet_embedding(test_loader, f_model)

        os.makedirs(path_to_cache, exist_ok=True)
        with open(os.path.join(path_to_cache, f"train_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(train_embeddings, file)

        with open(os.path.join(path_to_cache, f"valid_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(valid_embeddings, file)

        with open(os.path.join(path_to_cache, f"test_embeddings_{encoding}.data"), 'wb') as file:
            pickle.dump(test_embeddings, file)

        print(f"New embeddings are calculated and saved to {path_to_cache}")

    return train_embeddings, valid_embeddings, test_embeddings

