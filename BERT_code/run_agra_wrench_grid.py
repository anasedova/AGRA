import numpy as np
import torch

from wrench.endmodel.end_model_with_AGRA import EndClassifierModelWithAGRA
from wrench.dataset import load_dataset
from wrench.labelmodel.majority_voting import MajorityVoting
from snorkel.utils import probs_to_preds
from wrench.search import grid_search


if __name__ == '__main__':
    dataset_path = r'C:\Users\Lena\Documents\PhD - Preparation\GM Paper\New Code\datasets\datasets'
    data = 'youtube'
    extractor = 'tfidf'
    input_data_path = r"C:\Users\Lena\Documents\PhD - Preparation\GM Paper\New Code\datasets\datasets"  # /path/to/datasets"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = "MLP"

    # from grid_search.py
    # Todo: Add AGRA parameters here
    SEARCH_SPACE = {
        'LogReg': {
            'lr': np.logspace(-5, -1, num=5, base=10),
            'l2': np.logspace(-5, -1, num=5, base=10),
            'batch_size': [32, 128, 512],
        },
        'MLP': {
            'lr': np.logspace(-5, -1, num=5, base=10),
            'l2': np.logspace(-5, -1, num=5, base=10),
            'batch_size': [32, 128, 512],
            'hidden_size': [100],
            'dropout': [0.0],
        }
    }

    search_space = SEARCH_SPACE[backbone]

    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn=extractor
    )

    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=train_data)
    soft_labels_train = label_model.predict_proba(train_data)
    y_train = probs_to_preds(soft_labels_train)

    model = EndClassifierModelWithAGRA(
            batch_size=16,
            comp_batch_size=128,
            test_batch_size=512,
            comp_loss="CE",
            other=None,
            agra_threshold=0,
            n_steps=100,
            backbone=backbone,
            optimizer='Adam',
            optimizer_lr=0.01,
            optimizer_weight_decay=0.01
        )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 100
    n_repeats = 3
    searched_paras = grid_search(
        model,
        dataset_train=train_data,
        dataset_valid=valid_data,
        metric='acc',
        direction='auto',
        search_space=search_space,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=True,
        device=device,
    )

    #### Run end model: MLP
    # Todo: also add comp_batch_size, agra_threshold params to grid search
    model = EndClassifierModelWithAGRA(
        test_batch_size=512,
        comp_batch_size=128,
        n_steps=10000,
        backbone='MLP',
        optimizer='Adam',
        comp_loss="CE",
        other=None,
        agra_threshold=0,
        **searched_paras
    )

    model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device
    )

    acc = model.test(test_data, 'acc')
