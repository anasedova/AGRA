# from wrench
import numpy as np

from snorkel.utils import probs_to_preds
from wrench.wrench.labelmodel.majority_voting import MajorityVoting
from wrench.wrench.dataset import load_dataset


def prepare_data(dataset, dataset_type, path_to_data, extractor, model_name=None):
    """
    dataset: name of dataset, e.g. "sms", "youtube", ...
    path_to_data: storing location of the WRENCH datasets folder
    extractor: "tfidf" or "bert"
    return: X_train, y_train, X_dev, y_dev, X_test, y_test
    """

    train_data, valid_data, test_data = load_dataset(path_to_data, dataset, extract_feature=True, extract_fn=extractor,
                                                     dataset_type=dataset_type, model_name=model_name)

    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=train_data)
    soft_labels_train = label_model.predict_proba(train_data)
    hard_labels_train = probs_to_preds(soft_labels_train)

    # check which samples are mislabeled
    X_train = train_data.features
    gold_labels_train = train_data.labels
    mislabeled = np.where(hard_labels_train != gold_labels_train, 1, 0)
    noise_rate = sum(mislabeled) / len(mislabeled)

    # weights for class-weighted sampling
    weights = compute_sample_weights(hard_labels_train)

    return X_train, hard_labels_train, valid_data.features, np.array(valid_data.labels), test_data.features, \
           np.array(test_data.labels), np.array(gold_labels_train), np.array(mislabeled), weights, noise_rate


def compute_sample_weights(train_labels):
    """
    train_labels: weak training labels
    return: class-dependent weights for each sample
    """

    # determine the number of classes and samples
    num_classes = len(np.unique(train_labels))
    num_samples = len(train_labels)

    # count occurrences per label
    _, counts = np.unique(train_labels, return_counts=True)

    # initialize weights, source/cite
    weights = np.zeros(num_samples)
    for label in range(0, num_classes):
        weights[train_labels == label] = 1 / counts[label]

    return weights
