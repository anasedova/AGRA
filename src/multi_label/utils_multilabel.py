import os

import joblib
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


def set_seed(seed: int) -> None:
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AGRADataset(Dataset):

    def __init__(self, features, labels):
        self.len = labels.shape[0]

        if isinstance(features, np.ndarray) is False:
            self.features = torch.tensor(features.toarray()).float()

        else:
            self.features = torch.tensor(features).float()

        self.labels = torch.tensor(labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index].float()

    def __len__(self):
        return self.len


def load_image_dataset(
        # todo: add other part again
        data_path, dataset, encoding: str = "resnet50"
) -> Tuple[List[List], List[List], List[List]]:
    """ Load the train, valid and test sets for image dataset (currently: Cifar, CheXpert) """

    if os.path.exists(os.path.join(data_path, dataset, f"encoded_{encoding}")):
        train_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"train_embeddings_{encoding}.data"))
        valid_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"valid_embeddings_{encoding}.data"))
        test_embeddings = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"test_embeddings_{encoding}.data"))
        print(f"Embeddings are loaded from {os.path.join(data_path, dataset, f'encoded_{encoding}')}")

    else:
        raise ValueError('Specified embeddings not available.')

    return train_embeddings, valid_embeddings, test_embeddings

def load_image_labels(
        # todo: add other part again
        data_path, dataset, encoding: str = "resnet50"
) -> Tuple[List[List], List[List], List[List]]:
    """ Load the train, valid and test sets for image dataset (currently: Cifar, CheXpert) """

    train_labels = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"train_labels.data"))
    valid_labels = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"valid_labels.data"))
    test_labels = joblib.load(os.path.join(data_path, dataset, f"encoded_{encoding}", f"test_labels.data"))
    print(f"Labels are loaded from {os.path.join(data_path, dataset, f'encoded_{encoding}')}")

    return train_labels, valid_labels, test_labels


