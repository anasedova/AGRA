import argparse
import json
import os
import pprint
import time

import joblib
import numpy as np
from snorkel.utils import probs_to_preds

from _scr.utils import set_seed
from gridsearch import grid_search
from wrench.dataset import load_dataset
from wrench.endmodel.bert_model import BertClassifierModel
from wrench.endmodel.linear_model import LogRegModel
from wrench.evaluation import AverageMeter
from wrench.labelmodel.majority_voting import MajorityVoting

# from wrench.search import grid_search

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dataset", type=str, default='sms', choices=['youtube', 'sms', 'trec', 'yoruba', 'hausa'])
    parser.add_argument("--model", type=str, default='logreg', choices=['logreg', 'bert']) # or roberta-base
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    dataset_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'datasets')
    results_folder = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'results', 'baseline', args.dataset, args.model)
    os.makedirs(results_folder, exist_ok=True)
    output_file = open(os.path.join(results_folder, "test_performance.txt"), "w")

    if args.dataset in ['youtube', 'trec']:
        metric = 'acc'
    elif args.dataset == 'sms':
        metric = 'f1_binary'
    elif args.dataset in ['yoruba', 'hausa']:
        metric = 'f1_macro'
    else:
        raise ValueError('Dataset not available.')

    ### from wrench.search_space.py
    SEARCH_SPACE = {
        'logreg': {
            'lr': np.logspace(-5, -1, num=5, base=10),
            'l2': np.logspace(-5, -1, num=5, base=10),
            'batch_size': [32, 128, 512]
        },
        'roberta-base': {
            'lr': [5e-5, 3e-5, 2e-5],
            'batch_size': [32, 16]
        }
    }

    search_space = SEARCH_SPACE[args.model]

    # load training data
    if args.model == 'logreg':
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            args.dataset,
            dataset_type='TextDataset',
            extract_feature=True,
            extract_fn='tfidf'
        )
    elif args.model == 'bert':
        if args.data in ['yoruba', 'hausa']:
            model_name = 'bert-base-multilingual-cased'
        else:
            model_name = 'roberta-base'
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            args.dataset,
            dataset_type='TextDataset',
            extract_feature=False
        )
    else:
        raise ValueError('Model not implemented.')

    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=train_data)
    soft_labels_train = label_model.predict_proba(train_data)
    y_train = probs_to_preds(soft_labels_train)

    # initialize model
    if args.model == 'logreg':
        model = LogRegModel()
    elif args.model == 'bert':
        model = BertClassifierModel(model_name=model_name)
    else:
        raise ValueError('Model not implemented.')

    #### Search best hyper-parameters using validation set
    n_repeats = 3
    start_time = time.time()
    searched_paras, study = grid_search(
        model,
        dataset_train=train_data,
        y_train=y_train,
        dataset_valid=valid_data,
        metric=metric,
        direction='maximize',
        search_space=search_space,
        n_repeats=n_repeats,
        parallel=False,
        device=args.device,
    )
    end_time = time.time()

    print('Time spent on hyperparameter tuning:', end_time - start_time)

    if args.save is True:
        # save best parameters
        results_file = open(os.path.join(results_folder, 'searched_parameters.json'), "w")
        json.dump(searched_paras, results_file)

        # save study
        joblib.dump(study, os.path.join(results_folder, "optuna_study.pkl"))

    ### run on test
    n_repeats = 5
    meter = AverageMeter(names=['target'])
    for i in range(n_repeats):
        if args.model == 'logreg':
            model = LogRegModel(**searched_paras)
        elif args.model == 'bert':
            model = BertClassifierModel(model_name=model_name,
                                        **searched_paras)

        history = model.fit(dataset_train=train_data, y_train=y_train, dataset_valid=valid_data, metric=metric,
                                device=args.device)
        metric_value = model.test(test_data, metric)
        meter.update(target=metric_value)

    metrics = meter.get_results()
    pprint.pprint(metrics)

    if args.save is True:
        output_file.write("\t".join(["model", args.model]) + "\n")
        output_file.write("\t".join(["comp_loss", args.closs]) + "\n")
        output_file.write("\t".join(["weights", args.weights]) + "\n")
        output_file.write("\t".join(["alternative_label", str(args.other)]) + "\n")
        output_file.write("\t".join(["lr", str(searched_paras['lr'])]) + "\n")
        output_file.write("\t".join(["batch_size", str(searched_paras['batch_size'])]) + "\n")
        output_file.write("\t".join(["agra_threshold", str(searched_paras['agra_threshold'])]) + "\n")
        output_file.write("\t".join([metric, str(metrics['target'][0])]) + "\n")
        output_file.write("\t".join(["std", str(metrics['target'][1])]) + "\n")
        if args.model == 'logreg':
            output_file.write("\t".join(["l2", str(searched_paras['l2'])]) + "\n")