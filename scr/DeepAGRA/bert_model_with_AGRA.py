import copy
import logging
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from _scr.DeepAGRA.labels_recalculation_last_layer import calculate_label
from _scr.eval_plots import make_plots_gold
from _scr.utils import get_loss, get_statistics
from wrench.backbone import BackBone
from wrench.basemodel import BaseTorchClassModel
from wrench.dataset import BaseDataset
from wrench.utils import cross_entropy_with_probs, get_bert_model_class, get_bert_torch_dataset_class, \
    construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


class BertClassifierModelWithAGRA(BaseTorchClassModel):
    def __init__(self,
                 model_name: Optional[str] = 'bert-base-cased',
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
        self.hyperparas = {
            'model_name': model_name,
            'fine_tune_layers': fine_tune_layers,
            'lr': lr,
            'l2': l2,
            'max_tokens': max_tokens,
            'batch_size': batch_size,
            'real_batch_size': real_batch_size,
            'test_batch_size': test_batch_size,
            'comp_batch_size': comp_batch_size,  # maybe just use batch_size
            'comp_loss': comp_loss,
            'other': other,
            'agra_threshold': agra_threshold,
            'agra_weights': agra_weights,
            'ignore_index': ignore_index,
            'stats': stats,
            'storing_loc': storing_loc,
            'n_steps': n_steps
        }
        self.model: Optional[BackBone] = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            evaluation_step: Optional[int] = 10,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 10,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas

        n_steps = hyperparas['n_steps']
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size']:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']
        comp_batch_size = hyperparas['comp_batch_size']
        comp_loss = hyperparas['comp_loss']
        other = hyperparas['other']
        agra_threshold = hyperparas['agra_threshold']
        agra_weights = hyperparas['agra_weights']
        ignore_index = hyperparas['ignore_index']
        stats = hyperparas['stats']
        storing_loc = hyperparas['storing_loc']

        torch_dataset = get_bert_torch_dataset_class(dataset_train)(
            dataset_train, self.tokenizer, self.hyperparas['max_tokens'], n_data=n_steps * hyperparas['batch_size']
        )

        train_dataloader = DataLoader(
            torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=collate_fn
        )

        y_train = torch.Tensor(y_train).to(device)  # weak labels
        y_gold = torch.tensor(dataset_train.labels)  # gold labels

        if y_valid is None:
            y_valid = torch.tensor(dataset_valid.labels)  # valid labels

        # determine the number of classes
        if other is not None:
            num_classes = int(max(max(y_train), max(y_valid), other) + 1)
        else:
            num_classes = int(max(max(y_train), max(y_valid)) + 1)

        # initialize comparison loss
        comparison_criterion = get_loss(comp_loss, num_classes)

        if agra_weights is None:
            agra_weights = np.ones(len(dataset_train))
        agra_weights = torch.FloatTensor(agra_weights)

        model = get_bert_model_class(dataset_train)(n_class=num_classes, **hyperparas).to(device)
        self.model = model

        optimizer = AdamW(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        stats_all = []
        try:
            with trange(n_steps, desc=f"[TRAIN] {hyperparas['model_name']} Classifier", unit="steps",
                        disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                self.model.train()
                optimizer.zero_grad()
                for batch_num, batch in enumerate(train_dataloader):
                    # retrieve the weak labels
                    batch_idx = batch['ids'].to(device)
                    target = y_train[batch_idx]

                    # sample a comparison batch
                    comp_idx = list(WeightedRandomSampler(agra_weights, num_samples=comp_batch_size, replacement=False))
                    comp_dataset = torch.utils.data.Subset(torch_dataset, comp_idx)
                    comp_batch = next(iter(DataLoader(comp_dataset, batch_size=len(comp_idx), collate_fn=collate_fn)))
                    weak_comp_labels = y_train[comp_idx].long()

                    # calculate the labels that should be used for training
                    chosen_labels = calculate_label(
                        batch, target, comp_batch, weak_comp_labels,
                        copy.deepcopy(model), comparison_criterion,
                        other_class=other, threshold=agra_threshold,
                        ignore_index=ignore_index, device=device,
                        layer_aggregation=False, batch_num=batch_num
                    )

                    # compute statistics
                    if stats is True:
                        gold_train_labels = y_gold[batch_idx]
                        stats_summary = get_statistics(
                            target.to('cpu'), chosen_labels, gold_train_labels.long(), other, ignore_index
                        )
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
                        cnt += 1  # cnt here or outside

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                logger.info(info)
                                break
                            curr_val_metrics = {
                                'loss': loss.item(),
                                f'val_{metric}': metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step': self.best_step,
                            }
                            history[step] = curr_val_metrics
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if stats is True:
                # plot statistics up to best step
                stats_all = stats_all[:(self.best_step - 1)]
                make_plots_gold(stats_all, smoothing_length=10, other=other, storing_loc=storing_loc)

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history
