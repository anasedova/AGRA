import os

import joblib
import numpy as np
import random
import torch
from torch.utils.data import Dataset


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


def load_data(storing_loc_data):
    # train_data = pd.read_csv(os.path.join(storing_loc_data, "train_embeddings.csv"), header=None)
    train_data = joblib.load(os.path.join(storing_loc_data, "train_embeddings.lib"))

    X_train = train_data.iloc[:, 0:1280]
    y_train = train_data.iloc[:, 1280:train_data.shape[1]]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # dev_data = pd.read_csv(os.path.join(storing_loc_data, "valid_embeddings.csv"), header=None)
    dev_data = joblib.load(os.path.join(storing_loc_data, "valid_embeddings.lib"))
    X_dev = dev_data.iloc[:, 0:1280]
    y_dev = dev_data.iloc[:, 1280:train_data.shape[1]]

    X_dev = X_dev.to_numpy()
    y_dev = y_dev.to_numpy()

    # test_data = pd.read_csv(os.path.join(storing_loc_data, "test_embeddings.csv"), header=None)
    test_data = joblib.load(os.path.join(storing_loc_data, "test_embeddings.lib"))

    X_test = test_data.iloc[:, 0:1280]
    y_test = test_data.iloc[:, 1280:train_data.shape[1]]

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_dev, y_dev, X_test, y_test
