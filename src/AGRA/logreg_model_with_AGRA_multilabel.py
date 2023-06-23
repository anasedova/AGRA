#### adapted from https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/linear_model.py

import copy
import logging
import math
import warnings
from typing import Optional, Union, Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from wrench.dataset import BaseDataset, TorchDataset

from src.AGRA.labels_recalculation_logreg_multilabel import calculate_label_multilabel, MaskedBCELoss
from src.AGRA.logistic_regression import MaxEntNetwork
from src.utils import get_loss
from src.utils import set_seed

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
warnings.filterwarnings("ignore", category=UserWarning)         # ignore UserWarning
logger = logging.getLogger(__name__)


class LogRegModelWithAGRAMulti:
    def __init__(self,
                 num_classes: int,
                 include_bias: bool = False,
                 agra_threshold: float = 0,
                 ignore_index: int = -100,
                 storing_loc: str = None,
                 ):
        super().__init__()

        self.ignore_index = ignore_index
        self.agra_threshold = agra_threshold

        self.include_bias = include_bias

        self.num_classes = num_classes
        self.model = None
        self.best_metric_value = 0
        self.best_model_state = None
        self.storing_loc = storing_loc

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            lr: Optional[float] = 1e-4,
            l2: Optional[float] = 0.0,
            batch_size: Optional[int] = 32,
            comp_loss: str = "CE",
            num_epochs: int = 10,
            comp_batch_size: int = None,
            metric: Optional[Union[str, Callable]] = 'acc',
            verbose: Optional[bool] = True,
            seed: int = None
    ):

        if seed is not None:
            set_seed(seed)

        if not verbose:
            logger.setLevel(logging.ERROR)

        if comp_batch_size is None:
            comp_batch_size = batch_size

        train_dataloader = DataLoader(TorchDataset(dataset_train), batch_size=batch_size, shuffle=True)

        if y_train is None:
            y_train = torch.Tensor(dataset_train.labels) # this are the gold labels
        else:
            y_train = torch.Tensor(y_train)  # weak labels

        self.agra_weights = torch.FloatTensor(np.ones(dataset_train.features.shape[0])) # equal weights

        input_size = dataset_train.features.shape[1]  # size of feature vector

        # initialize the model
        self.model = MaxEntNetwork(input_size, self.num_classes).to(device)

        # initialize loss & optimizer
        comparison_criterion, loss_type = get_loss(comp_loss, self.num_classes, multilabel=True)
        update_criterion = MaskedBCELoss(self.ignore_index, self.num_classes)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        self.best_epoch = 0
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                # retrieve the weak labels
                train_samples = batch["ids"]
                weak_train_labels = y_train[train_samples].long()

                # sample a comparison batch
                idx = list(WeightedRandomSampler(self.agra_weights, num_samples=comp_batch_size, replacement=False))
                comp_dataset = dataset_train.create_subset(idx)
                comp_batch = next(iter(DataLoader(TorchDatasetMulti(comp_dataset), batch_size=len(idx))))
                weak_comp_labels = y_train[idx].long()

                # calculate the labels that should be used for training
                chosen_labels = calculate_label_multilabel(batch, weak_train_labels, comp_batch, weak_comp_labels,
                                                copy.deepcopy(self.model), comparison_criterion=comparison_criterion,
                                                ignore_index=self.ignore_index,
                                                threshold=self.agra_threshold,
                                                include_bias=self.include_bias,
                                                loss_type=loss_type)

                # use masked loss
                outputs = torch.sigmoid(self.model(batch["features"].to(device)))
                loss = update_criterion(outputs, chosen_labels.float().to(device))
                loss.backward()
                optimizer.step()

            # for selection of best epoch on validation set
            print('Epoch', self.epoch, 'finished.')
            _ = self.test_chexpert(dataset_valid, batch_size, metric)

        self.model.load_state_dict(self.best_model_state)

        return self.best_metric_value

    def test_chexpert(self, dataset_valid, batch_size, metric, test_mode: bool = False):
        """
        test function used for chexpert
        """
        # validation
        valid_data = TorchDataset(dataset_valid)

        dev_prediction_probas = []

        self.model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(dataset_valid):
                probas = torch.sigmoid(self.model(torch.tensor(data).to(device)))
                dev_prediction_probas.append(probas.cpu().numpy())

        # remove the labels for Lung Lesion, Pleural Other and Fracture
        labels = np.delete(valid_data.labels, [3, 10, 11], 1)
        prediction_probs = np.delete(np.asarray(dev_prediction_probas), [3, 10, 11], 1)

        if metric == 'macro_auc':
            metric_value = roc_auc_score(labels, prediction_probs)
        else:
            raise ValueError('Given metric not implemented.')

        if test_mode is False:
            # record metrics for choosing best epoch on validation
            if metric_value > self.best_metric_value:
                self.best_metric_value = metric_value
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = self.epoch
            self.model.train()

        return metric_value

    def test(self, dataset_valid, batch_size, metric, test_mode: bool = False):
        """
        general test function
        """
        # validation
        self.model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            valid_dataloader = DataLoader(TorchDataset(dataset_valid), batch_size=batch_size)
            for batch in valid_dataloader:
                ex = batch['features'].to(device)
                labels = batch['labels'].to(device)
                preds = torch.sigmoid(self.model(ex))
                all_preds.append(preds.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            metric_value = roc_auc_score(all_labels, all_preds)

            if test_mode is False:
                # record metrics for choosing best epoch on validation
                if metric_value > self.best_metric_value:
                    self.best_metric_value = metric_value
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = self.epoch
                self.model.train()

        return metric_value

class TorchDatasetMulti(Dataset):
    def __init__(self, dataset: BaseDataset, n_data: Optional[int] = 0):
        self.features = dataset.features
        self.labels = dataset.labels
        self.weak_labels = dataset.weak_labels
        self.data = dataset.examples
        n_data_ = len(self.data)
        self.n_data_ = n_data_
        if n_data > 0:
            self.n_data = math.ceil(n_data / n_data_) * n_data_
        else:
            self.n_data = n_data_

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        d = {
            'ids'        : idx,
            'labels'     : self.labels[idx],
            'weak_labels': self.weak_labels[idx],
            'data'       : self.data[idx],
        }
        if self.features is not None:
            d['features'] = self.features[idx]
        return d