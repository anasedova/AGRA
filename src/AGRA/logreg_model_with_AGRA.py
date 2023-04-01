#### adapted from https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/linear_model.py

import copy
import logging
import warnings
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from src.AGRA.labels_recalculation_logreg import calculate_label
from src.multi_label.logistic_regression import MaxEntNetwork
from src.utils import get_loss
from wrench.backbone import BackBone
from wrench.dataset import BaseDataset, TorchDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
warnings.filterwarnings("ignore", category=UserWarning)         # ignore UserWarning
logger = logging.getLogger(__name__)


class LogRegModelWithAGRA:
    def __init__(self,
                 num_classes: int,
                 include_bias: bool = True,
                 adaptive_threshold: bool = True,
                 other: int = None,
                 agra_threshold: float = 0,
                 agra_weights: torch.tensor = None,
                 ignore_index: int = -100
                 ):
        super().__init__()

        self.other = other
        self.ignore_index = ignore_index
        self.agra_threshold = agra_threshold
        self.agra_weights = agra_weights

        self.include_bias = include_bias
        self.adaptive_threshold = adaptive_threshold

        self.num_classes = num_classes
        self.model: Optional[BackBone] = None
        self.best_metric_value = 0
        self.best_model_state = None

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,

            lr: Optional[float] = 1e-4,
            l2: Optional[float] = 0.0,
            batch_size: Optional[int] = 32,
            comp_loss: str = "CE",
            num_epochs: int = 10,

            comp_batch_size: int = None,
            metric: Optional[Union[str, Callable]] = 'acc',
            tolerance: Optional[float] = -1.0,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        if comp_batch_size is None:
            comp_batch_size = batch_size

        train_dataloader = DataLoader(TorchDataset(dataset_train), batch_size=batch_size, shuffle=True)

        if y_train is None:
            y_train = torch.Tensor(dataset_train.weak_labels)
        else:
            y_train = torch.Tensor(y_train)  # weak labels
        # y_gold = torch.Tensor(dataset_train.labels)  # gold labels

        if self.agra_weights is None:
            self.agra_weights = np.ones(dataset_train.features.shape[0])
        self.agra_weights = torch.FloatTensor(self.agra_weights)

        input_size = dataset_train.features.shape[1]  # size of feature vector

        # initialize the model
        self.model = MaxEntNetwork(input_size, self.num_classes).to(device)

        # initialize loss & optimizer
        comparison_criterion = get_loss(comp_loss, self.num_classes)
        update_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_index)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()

                # retrieve the weak labels
                train_samples = batch["ids"]
                weak_train_labels = y_train[train_samples].long()

                # sample a comparison batch
                idx = list(WeightedRandomSampler(self.agra_weights, num_samples=comp_batch_size, replacement=False))
                comp_dataset = dataset_train.create_subset(idx)
                comp_batch = next(iter(DataLoader(TorchDataset(comp_dataset), batch_size=len(idx))))
                weak_comp_labels = y_train[idx].long()

                # calculate the labels that should be used for training
                chosen_labels = calculate_label(batch, weak_train_labels, comp_batch, weak_comp_labels,
                                                copy.deepcopy(self.model), comparison_criterion=comparison_criterion,
                                                ignore_index=self.ignore_index, other_class=self.other,
                                                threshold=self.agra_threshold,
                                                include_bias=self.include_bias,
                                                adaptive_threshold=self.adaptive_threshold)

                # remove ignored samples and their model outputs
                ignore_samples = np.where(chosen_labels == self.ignore_index)
                chosen_labels = np.delete(chosen_labels, ignore_samples)
                chosen_labels = torch.Tensor(chosen_labels).to(device)
                if len(chosen_labels) > 0:
                    outputs = self.model(batch["features"].to(device))
                    remove_mask = np.zeros([len(outputs)])
                    remove_mask[ignore_samples] = 1
                    outputs = outputs[remove_mask != 1]

                    # compute loss only with the selected examples
                    loss = update_criterion(outputs, chosen_labels)
                    loss.backward()
                    optimizer.step()

            _ = self.test(dataset_valid, batch_size, metric)
            # print(f"Epoch {epoch} \t valid metric: {metric_value}")

        # logger.info(f"The best metric: {self.best_metric_value}, the model state will be loaded...")
        self.model.load_state_dict(self.best_model_state)

    def test(self, dataset_valid, batch_size, metric):
        # validation
        self.model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size)
            for batch, labels in valid_dataloader:
                batch, labels = batch.to(device), labels.to(device)
                preds = torch.argmax(self.model(batch), dim=1)
                all_preds.append(preds.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            report = classification_report(y_true=all_labels, y_pred=all_preds, output_dict=True)
            if metric == "acc":
                metric_value = report["accuracy"]
            elif metric == "f1":
                metric_value = report["macro avg"]
            else:
                raise ValueError(f"Unknown metric {metric}")

            if metric_value > self.best_metric_value:
                self.best_metric_value = metric_value
                self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.model.train()
        return metric_value

