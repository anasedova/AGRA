#### adapted from https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/linear_model.py

import copy
import logging
import math
import warnings
from typing import Optional, Union, Callable

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from wrench.dataset import BaseDataset, TorchDataset

from src.AGRA.labels_recalculation_logreg import calculate_label
from src.AGRA.logistic_regression import MaxEntNetwork
from src.eval_plots import get_statistics, make_plots_gold
from src.utils import get_loss, set_seed

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
warnings.filterwarnings("ignore", category=UserWarning)         # ignore UserWarning
logger = logging.getLogger(__name__)


class LogRegModelWithAGRA:
    def __init__(self,
                 num_classes: int,
                 include_bias: bool = False,
                 adaptive_threshold: bool = True,
                 other: int = None,
                 agra_threshold: float = 0,
                 agra_weights: torch.tensor = None,
                 ignore_index: int = -100,
                 collect_stats: bool = False,
                 storing_loc: str = None,
                 smooth: int = 1
                 ):
        super().__init__()

        self.other = other
        self.ignore_index = ignore_index
        self.agra_threshold = agra_threshold
        self.agra_weights = agra_weights

        self.include_bias = include_bias
        self.adaptive_threshold = adaptive_threshold

        self.num_classes = num_classes
        self.model = None
        self.best_metric_value = 0
        self.best_model_state = None
        self.collect_stats = collect_stats
        self.storing_loc = storing_loc
        self.smooth = smooth

    def fit(
            self,
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

        # initialize the model
        input_size = dataset_train.features.shape[1]  # size of feature vector
        self.model = MaxEntNetwork(input_size, self.num_classes).to(device)

        if comp_batch_size is None:
            comp_batch_size = batch_size

        train_dataloader = DataLoader(TorchDataset(dataset_train), batch_size=batch_size, shuffle=True)

        if y_train is None:
            y_train = torch.Tensor(dataset_train.labels) # this are the gold labels
        else:
            y_train = torch.Tensor(y_train)  # weak labels
        y_gold = torch.Tensor(dataset_train.labels)  # gold labels
        num_samples = len(y_train)

        if self.agra_weights is None:
            self.agra_weights = np.ones(dataset_train.features.shape[0])
        self.agra_weights = torch.FloatTensor(self.agra_weights)

        # initialize loss & optimizer
        comparison_criterion, loss_type = get_loss(comp_loss, self.num_classes, multilabel=False)
        update_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        stats = []
        self.best_epoch = 0
        for epoch in range(num_epochs):
            print('Epoch', epoch)
            self.epoch = epoch
            self.model.train()
            for batch in train_dataloader:

                # retrieve the weak labels
                train_samples = batch["ids"]
                weak_train_labels = y_train[train_samples].long()
                gold_train_labels = y_gold[train_samples].long()

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
                                                adaptive_threshold=self.adaptive_threshold,
                                                loss_type=loss_type)
                if self.collect_stats is True:
                    stats_gold = get_statistics(weak_labels=weak_train_labels, chosen_labels=chosen_labels, gold_labels=gold_train_labels, other=self.other, ignore_index=self.ignore_index)
                    stats.append(stats_gold)

                # remove ignored samples and their model outputs
                ignore_samples = np.where(chosen_labels == self.ignore_index)
                chosen_labels = np.delete(chosen_labels, ignore_samples)
                chosen_labels = torch.Tensor(chosen_labels).to(device)
                optimizer.zero_grad()
                if len(chosen_labels) > 0:
                    outputs = self.model(batch["features"].to(device))
                    remove_mask = np.zeros([len(outputs)])
                    remove_mask[ignore_samples] = 1
                    outputs = outputs[remove_mask != 1]

                    # compute loss only with the selected examples
                    loss = update_criterion(outputs, chosen_labels)
                    loss.backward()
                    optimizer.step()

            # for selection of best epoch on validation set
            _ = self.test(dataset_valid, batch_size, metric)

        self.model.load_state_dict(self.best_model_state)

        if self.collect_stats is True:
            stats_gold_plot = stats[0:(math.ceil(num_samples / batch_size)*(self.best_epoch + 1))]
            make_plots_gold(stats_gold_plot, storing_loc=self.storing_loc, smoothing_length=self.smooth)

        return self.best_metric_value

    def test(self, dataset_valid, batch_size, metric, test_mode: bool = False):
        self.model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            valid_dataloader = DataLoader(TorchDataset(dataset_valid), batch_size=batch_size)
            for batch in valid_dataloader:
                ex = batch['features'].to(device)
                labels = batch['labels'].to(device)
                preds = torch.argmax(self.model(ex), dim=1)
                all_preds.append(preds.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            report = classification_report(y_true=all_labels, y_pred=all_preds, output_dict=True)
            if metric == "acc":
                metric_value = report["accuracy"]
            elif metric == "f1_macro":
                metric_value = report["macro avg"]["f1-score"]
            elif metric == "f1_binary":
                metric_value = report["1"]["f1-score"]
            else:
                raise ValueError(f"Unknown metric {metric}")

            if test_mode is False:
                # record metrics for choosing best epoch on validation
                if metric_value > self.best_metric_value:
                    self.best_metric_value = metric_value
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = self.epoch
                self.model.train()

        return metric_value

    def test_no_batches(self, dataset_valid, batch_size, metric, test_mode=False):
        """
        This version was used for running the ECML experiments, for large datasets the "test" function is more suitable
        """
        self.model.eval()
        valid_data = TorchDataset(dataset_valid)
        with torch.no_grad():
            dev_prediction_probas = self.model(torch.tensor(valid_data.features).to(device))
            dev_predictions = torch.argmax(dev_prediction_probas, dim=1)
        report = classification_report(y_true=torch.tensor(valid_data.labels), y_pred=dev_predictions.cpu(), output_dict=True)
        if metric == "acc":
            metric_value = report["accuracy"]
        elif metric == "f1_macro":
            metric_value = report["macro avg"]["f1-score"]
        elif metric == "f1_binary":
            metric_value = report["1"]["f1-score"]
        else:
            raise ValueError(f"Unknown metric {metric}")
        if test_mode is False:
            # record metrics for choosing best epoch on validation
            if metric_value >= self.best_metric_value:
                self.best_metric_value = metric_value
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = self.epoch
            self.model.train()

        return metric_value
