import copy
import logging
from typing import Any, Optional, Union, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup

from src.eval_plots import make_plots_gold
from src.labels_recalculation import calculate_label
from src.utils import get_loss
from wrench.basemodel import BaseTorchClassModel
from wrench.dataset import BaseDataset
from wrench.evaluation import METRIC, metric_to_direction
from wrench.utils import construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


class ResNetClassifierModelWithDeepAGRA(BaseTorchClassModel):
    def __init__(self,
                 model_name: Optional[str] = 'resnet50',
                 lr: Optional[float] = 3e-5,
                 l2: Optional[float] = 0.0,
                 max_tokens: Optional[int] = 512,
                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 comp_batch_size: Optional[int] = 16,
                 comp_loss: str = "CE",
                 other: int = None,
                 agra_threshold: float = 0,
                 agra_weights: torch.tensor = None,
                 ignore_index: int = -100,
                 stats: bool = False,
                 storing_loc: str = None,
                 n_steps: Optional[int] = 10000,
                 fine_tune_layers: Optional[int] = -1
                 ):
        super().__init__()

        self.model_name = model_name
        self.lr = lr
        self.l2 = l2
        self.n_steps = n_steps
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.comp_batch_size = batch_size if comp_batch_size is None else comp_batch_size
        self.real_batch_size = batch_size \
            if real_batch_size is None or (real_batch_size == -1 or batch_size < real_batch_size) else real_batch_size
        self.test_batch_size = batch_size if test_batch_size is None else test_batch_size

        self.accum_steps = self.batch_size // self.real_batch_size
        self.comp_batch_size = comp_batch_size
        self.test_batch_size = test_batch_size
        self.comp_loss = comp_loss
        self.other = other
        self.agra_threshold = agra_threshold
        self.agra_weights = agra_weights
        self.ignore_index = ignore_index
        self.stats = stats
        self.storing_loc = storing_loc
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.fine_tune_layers = fine_tune_layers

        self.direction, self.metric_fn = None, None

        self.hyperparas = self._initialize_hyperparams_dict()

        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def fit(self,
            dataset_train: Dataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Tuple[BaseDataset, ImageFolder]] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            evaluation_step: Optional[int] = 10,
            metric: Optional[Union[str, Callable]] = 'acc',
            patience: Optional[int] = 10,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            class_weights: np.array = None,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        self.define_metric(metric)
        self._update_hyperparas(**kwargs)

        y_train = torch.Tensor(y_train).to(self.device)  # weak labels

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # Don't train on last batch: could be 1 noisy example. % todo: clarify whether we need it
        )

        if dataset_valid:
            val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=False)
            if y_valid is None:
                y_valid = np.array(dataset_valid.labels)  # valid labels

        # determine the number of classes
        if self.other is not None:
            num_classes = int(max(max(y_train), self.other) + 1)
        else:
            num_classes = int(max(y_train) + 1)

        if self.agra_weights is None:
            self.agra_weights = np.ones(len(dataset_train))
        self.agra_weights = torch.FloatTensor(self.agra_weights)

        # initialize the model to be trained
        self.model = models.__dict__[self.model_name](num_classes=num_classes).to(device)

        # initialize comparison loss
        comparison_criterion = get_loss(self.comp_loss, num_classes)
        # initialize update loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, self.direction, patience, tolerance)

        history = {}
        last_step_log = {}
        stats_all = []
        try:
            with trange(self.n_steps, desc=f"[TRAIN] {self.model_name} Classifier", unit="steps",
                        disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                self.model.train()
                optimizer.zero_grad()

                for batch_num, (batch, labels) in enumerate(train_dataloader):

                    # if batch is size 1, skip because batch-norm will fail
                    if len(batch) <= 1:
                        continue

                    batch, labels = batch.to(self.device), labels.to(self.device)

                    # select samples to be ignored and calculate the new labels
                    ignore_samples, chosen_labels = self.choose_samples_n_labels(
                        batch, labels, copy.deepcopy(self.model), dataset_train, comparison_criterion
                    )

                    if len(chosen_labels) <= 0:
                        continue

                    # calculate predictions
                    outputs = self.model(batch)

                    # remove ignored samples
                    outputs = remove_ignored(outputs, ignore_samples)

                    # compute loss only with the selected examples
                    loss = criterion(outputs, chosen_labels)
                    cnt += 1  # cnt here or outside

                    if cnt % self.accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step, dataloader=val_loader)
                            if early_stop_flag:
                                logger.info(info)
                                break
                            history[step] = {
                                'loss': loss.item(),
                                f'val_{metric}': metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step': self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= self.n_steps:
                            break

                    scheduler.step()

                    # # compute statistics
                    # if stats is True:
                    #     gold_train_labels = y_gold[batch_idx]
                    #     stats_summary = get_statistics(
                    #         target.to('cpu'), chosen_labels, gold_train_labels.long(), other, ignore_index
                    #     )
                    #     stats_all.append(stats_summary)

                    # todo: check what we should take from the cifar-cleanlab code

                    # # measure accuracy and record loss
                    # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    # losses.update(loss.item(), input.size(0))
                    # top1.update(acc1[0], input.size(0))
                    # top5.update(acc5[0], input.size(0))
                    #
                    # if i % args.print_freq == 0:
                    #     print('Epoch: [{0}][{1}/{2}]\t'
                    #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    #         epoch, i, len(train_loader), batch_time=batch_time,
                    #         data_time=data_time, loss=losses, top1=top1, top5=top5))

            if self.stats is True:
                # plot statistics up to best step
                stats_all = stats_all[:(self.best_step - 1)]
                make_plots_gold(stats_all, smoothing_length=10, other=self.other, storing_loc=self.storing_loc)

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history

    def _calc_valid_metric(self, dataloader, **kwargs):
        return self.test(dataloader)

    def _init_valid_dataloader(self, dataset_valid):
        return torch.utils.data.DataLoader(dataset_valid, batch_size=self.test_batch_size, shuffle=False)

    # def _finalize(self):
    #     self.model.load_state_dict(self.best_model)
    #     del self.best_model
    #     del self.valid_dataloader
    #     del self.y_valid

    def test(self, dataloader, **kwargs):
        predictions_list, label_list = [], []
        with torch.no_grad():
            for data, labels in dataloader:
                output = self.model(data.to(self.device))
                predictions = output.detach().cpu().numpy()
                predictions_list.append(predictions)
                label_list.append(labels.detach().cpu().numpy())

        predictions = np.concatenate(predictions_list)
        gold_labels = np.squeeze(np.hstack(label_list))
        return self.metric_fn(gold_labels, predictions)

    def choose_samples_n_labels(
            self, batch, labels, model, dataset_train, comparison_criterion
    ) -> Tuple[np.array, np.array]:
        # sample a comparison batch
        comp_idx = list(WeightedRandomSampler(self.agra_weights, num_samples=self.comp_batch_size, replacement=False))
        comp_dataset = torch.utils.data.Subset(dataset_train, comp_idx)
        comp_batch = next(iter(DataLoader(comp_dataset, batch_size=len(comp_idx))))

        comp_samples = comp_batch[0].to(self.device)
        comp_labels = comp_batch[1].to(self.device)

        # calculate the labels that should be used for training
        chosen_labels = calculate_label(
            batch,
            labels,
            comp_samples, comp_labels,
            model,
            comparison_criterion,
            other_class=self.other,
            threshold=self.agra_threshold,
            ignore_index=self.ignore_index,
            layer_aggregation=False,
            device=self.device
        )

        # remove ignored samples and their model outputs
        ignore_samples = np.where(chosen_labels == self.ignore_index)
        chosen_labels = np.delete(chosen_labels, ignore_samples).to(self.device)

        return ignore_samples, chosen_labels

    def define_metric(self, metric: Optional[Union[str, Callable]]) -> None:
        self.direction = metric_to_direction(metric)
        # initialize metric
        if isinstance(metric, Callable):
            metric_fn = metric
        else:
            metric_fn = METRIC[metric]
        self.metric_fn = metric_fn

    def _initialize_hyperparams_dict(self):
        return {
            'model_name': self.model_name,
            'fine_tune_layers': self.fine_tune_layers,
            'lr': self.lr,
            'l2': self.l2,
            'max_tokens': self.max_tokens,
            'batch_size': self.batch_size,
            'real_batch_size': self.real_batch_size,
            'test_batch_size': self.test_batch_size,
            'comp_batch_size': self.comp_batch_size,  # maybe just use batch_size
            'comp_loss': self.comp_loss,
            'other': self.other,
            'agra_threshold': self.agra_threshold,
            'agra_weights': self.agra_weights,
            'ignore_index': self.ignore_index,
            'stats': self.stats,
            'storing_loc': self.storing_loc,
            'n_steps': self.n_steps
        }


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
    specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def remove_ignored(outputs, ignore_samples):
    remove_mask = np.zeros([len(outputs)])
    remove_mask[ignore_samples] = 1
    return outputs[remove_mask != 1]
