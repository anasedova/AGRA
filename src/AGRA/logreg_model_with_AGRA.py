#### adapted from https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/linear_model.py
import logging
from typing import Any, Optional, Union, Callable, Dict

import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import get_linear_schedule_with_warmup

from src.AGRA.labels_recalculation_logreg import calculate_label
from src.utils import get_statistics, get_loss
from wrench.backbone import BackBone, LogReg
from wrench.basemodel import BaseTorchClassModel
from wrench.dataset import BaseDataset, TorchDataset
from wrench.utils import cross_entropy_with_probs
# from _src.labels_recalculation_logreg import calculate_label
from src.eval_plots import make_plots_gold

logger = logging.getLogger(__name__)


class LogRegModelWithAGRA(BaseTorchClassModel):
    def __init__(self,
                 lr: Optional[float] = 1e-4,
                 l2: Optional[float] = 0.0,
                 batch_size: Optional[int] = 128,
                 comp_batch_size: Optional[int] = None,
                 comp_loss: str = "CE",
                 other: int = None,
                 agra_threshold: float = 0,
                 agra_weights: torch.tensor = None,
                 ignore_index: int = -100,
                 stats: bool = False,
                 storing_loc: str = None,
                 test_batch_size: Optional[int] = 512,
                 n_steps: Optional[int] = 100000,
                 binary_mode: Optional[bool] = False
                 ):
        super().__init__()
        self.hyperparas = {
            'lr'             : lr,
            'l2'             : l2,
            'batch_size'     : batch_size,
            'comp_batch_size': comp_batch_size,
            'comp_loss'      : comp_loss,
            'other'          : other,
            'agra_threshold' : agra_threshold,
            'agra_weights'   : agra_weights,
            'ignore_index'   : ignore_index,
            'stats'          : stats,
            'storing_loc'    : storing_loc,
            'test_batch_size': test_batch_size,
            'n_steps'        : n_steps,
            'binary_mode'    : binary_mode,
        }
        self.model: Optional[BackBone] = None

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any) -> Dict:

        if not verbose:
            logger.setLevel(logging.ERROR)

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas

        n_steps = hyperparas['n_steps']
        train_dataloader = DataLoader(TorchDataset(dataset_train, n_data=n_steps * hyperparas['batch_size']),
                                      batch_size=hyperparas['batch_size'], shuffle=True)

        batch_size = hyperparas['batch_size']
        comp_batch_size = hyperparas['comp_batch_size']
        if comp_batch_size is None:
            comp_batch_size = batch_size
        comp_loss = hyperparas['comp_loss']
        other = hyperparas['other']
        agra_threshold = hyperparas['agra_threshold']
        agra_weights = hyperparas['agra_weights']
        ignore_index = hyperparas['ignore_index']
        stats = hyperparas['stats']
        storing_loc = hyperparas['storing_loc']

        if y_train is None:
            y_train = torch.Tensor(dataset_train.weak_labels).to(device)
        else:
            y_train = torch.Tensor(y_train).to(device)  # weak labels
        y_gold = torch.Tensor(dataset_train.labels)     # gold labels

        if agra_weights is None:
            # agra_weights = np.ones(len(dataset_train))
            agra_weights = np.ones(dataset_train.features.shape[0])
        agra_weights = torch.FloatTensor(agra_weights)

        # determine the number of classes
        if other is not None:
            num_classes = int(max(other, max(dataset_train.labels), max(dataset_valid.labels))) + 1
        else:
            num_classes = int(max(max(dataset_train.labels), max(dataset_valid.labels))) + 1

        input_size = dataset_train.features.shape[1]    # size of feature vector

        # initialize comparison loss
        comparison_criterion = get_loss(comp_loss, num_classes)

        model = LogReg(
            input_size=input_size, n_class=num_classes, binary_mode=hyperparas['binary_mode'],
        ).to(device)
        self.model = model

        optimizer = optim.Adam(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, metric=metric)

        history = {}
        last_step_log = {}
        stats_all = []
        try:
            with trange(n_steps, desc="[TRAIN] Linear Classifier", unit="steps", disable=not verbose, ncols=150,
                        position=0, leave=True) as pbar:
                model.train()
                step = 0
                for batch in train_dataloader:
                    step += 1
                    optimizer.zero_grad()

                    # retrieve the weak labels
                    train_samples = batch["ids"]
                    weak_train_labels = y_train[train_samples].long()

                    # sample a comparison batch
                    idx = list(WeightedRandomSampler(agra_weights, num_samples=comp_batch_size, replacement=False))
                    comp_dataset = dataset_train.create_subset(idx)
                    comp_batch = next(iter(DataLoader(TorchDataset(comp_dataset), batch_size=len(idx))))
                    weak_comp_labels = y_train[idx].long()

                    # calculate the labels that should be used for training
                    chosen_labels = calculate_label(batch, weak_train_labels, comp_batch, weak_comp_labels,
                                                    copy.deepcopy(model), comparison_criterion,
                                                    other_class=other, threshold=agra_threshold,
                                                    ignore_index=ignore_index, device=device)

                    # compute statistics
                    if stats is True:
                        gold_train_labels = y_gold[train_samples]
                        # correctly_removed, falsely_removed, falsely_kept, correctly_kept, correctly_corrected, falsely_corrected
                        stats_summary = get_statistics(weak_train_labels.to('cpu'), chosen_labels, gold_train_labels.long(),
                                               other, ignore_index)
                        stats_all.append(stats_summary)

                    # remove ignored samples and their model outputs
                    ignore_samples = np.where(chosen_labels == ignore_index)
                    chosen_labels = np.delete(chosen_labels, ignore_samples).to(device)
                    if len(chosen_labels) > 0:
                        outputs = model(batch)
                        remove_mask = np.zeros([len(outputs)])
                        remove_mask[ignore_samples] = 1
                        outputs = outputs[remove_mask != 1]

                        # compute loss only with the selected examples
                        loss = cross_entropy_with_probs(outputs, chosen_labels, reduction='mean')
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    if valid_flag and step % evaluation_step == 0:
                        metric_value, early_stop_flag, info = self._valid_step(step)
                        if early_stop_flag:
                            logger.info(info)
                            break

                        history[step] = {
                            'loss'              : loss.item(),
                            f'val_{metric}'     : metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step'         : self.best_step,
                        }
                        last_step_log.update(history[step])

                    last_step_log['loss'] = loss.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps:
                        break

            if stats is True:
                # plot statistics up to best step
                stats_all = stats_all[:(self.best_step-1)]
                make_plots_gold(stats_all, smoothing_length=10, other=other, storing_loc=storing_loc)

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history
