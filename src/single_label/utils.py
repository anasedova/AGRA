import os
import random
import logging
from typing import Tuple, List, Union

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset

logger = logging.getLogger(__name__)


def process_knodle_data_new(
        path_to_train: str,
        path_to_train_z,
        path_to_train_t,
        path_to_dev: str,
        path_to_test: str = None,
        no_relation_index: int = None,
        filter_ties: bool = False,
        return_raw: bool = False,
        default_vectorizer: str = None,
):
    # read in training csv
    df_train = pd.read_csv(path_to_train, sep='\t')
    tokens_train = pd.DataFrame(df_train["sample"])
    gold_labels = df_train["gold_label"]

    # create T and Z matrices
    z = joblib.load(path_to_train_z)
    t = joblib.load(path_to_train_t)

    # compute "probabilities" of each label by counting the rule matches for each class and dividing by the total
    # number of rule matches
    noisy_y_train_probs = z_t_matrices_to_majority_vote_probs(z, t)

    # identify the indices of samples where no rules matched
    zeros = np.where(noisy_y_train_probs.sum(axis=1) == 0)[0]
    non_zeros = np.where(noisy_y_train_probs.sum(axis=1) != 0)[0]
    logger.info(f"Samples without matches: {len(zeros)} out of {len(noisy_y_train_probs)}")

    # assign weak labels to all samples
    noisy_y_train = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=noisy_y_train_probs,
                                        filter_ties=filter_ties)

    # filter out samples without matches if specified
    if no_relation_index is None:
        tokens_train, noisy_y_train, z, gold_labels = filter_empty_probabilities(tokens_train, noisy_y_train, zeros,
                                                                                 non_zeros, gold_labels, z)

    # otherwise replace them with a specified label
    else:
        noisy_y_train[zeros] = no_relation_index

    # filter out ties if specified
    if filter_ties is True:
        tokens_train = tokens_train[noisy_y_train != -100]
        z = z[noisy_y_train != -100, :]
        gold_labels = gold_labels[noisy_y_train != -100]
        noisy_y_train = noisy_y_train[noisy_y_train != -100]

    # add the noisy labels to dataframe
    tokens_train["label"] = noisy_y_train

    # pre-process the sentences
    tokens_rels_train = [(x[0].replace("\ufeff", "").split(), str(x[1])) for x in tokens_train.values]

    # for BERT preprocessing just replace the weird character, don't split yet, the BERT tokenizer will do that
    if return_raw is True:
        tokens_rels_train_raw = [x[0].replace("\ufeff", "") for x in tokens_train.values]

    # apply the specified encoding (tfidf or count) to the training inputs
    # apply label encoder to labels to make sure they are of the form 0,1,..,K
    train_sentences_ds, train_labels_ds, vectorizer, label_encoder = \
        vectorize_data(tokens_rels_train, default_vectorizer=default_vectorizer)

    # check which samples are mislabeled
    # apply label encoder to the gold labels
    gold_labels = label_encoder.transform(
        [str(g) for g in gold_labels])  # get gold labels in same format as training labels
    mislabeled = np.zeros(train_sentences_ds.shape[0])
    mislabeled[np.where(train_labels_ds != gold_labels)] = 1

    # print the overall percentage of noisy labels in the dataset
    noise_rate = sum(mislabeled) / len(mislabeled)
    logger.info(f"Noise rate: {noise_rate}")

    # read in the dev data
    tokens_rels_dev = pd.read_csv(path_to_dev, sep='\t')

    # for BERT encodings
    if return_raw is True:
        tokens_rels_dev_raw = [x[0].replace("\ufeff", "") for x in tokens_rels_dev.values]

    # preprocess the dev data with vectorizer and label encoder fit to train data
    tokens_rels_dev = [(x[0].replace("\ufeff", "").split(), str(x[1])) for x in tokens_rels_dev.values]
    dev_sentences, dev_labels, _, _ = \
        vectorize_data(tokens_rels_dev, vectorizer=vectorizer, label_encoder=label_encoder)

    # if there is a third data part also repeat for this one
    if path_to_test is not None:
        tokens_rels_test = pd.read_csv(path_to_test, sep='\t')
        if return_raw is True:
            tokens_rels_test_raw = [x[0].replace("\ufeff", "") for x in tokens_rels_test.values]
        tokens_rels_test = [(x[0].replace("\ufeff", "").split(), str(x[1])) for x in tokens_rels_test.values]
        test_sentences, test_labels, _, _ = \
            vectorize_data(tokens_rels_test, vectorizer=vectorizer, label_encoder=label_encoder)

        logger.info(f"Other class id for labeling: {no_relation_index}")
        if return_raw is False:
            return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, \
                   torch.tensor(gold_labels), torch.tensor(mislabeled), no_relation_index, noise_rate
        else:
            return tokens_rels_train_raw, train_labels_ds, tokens_rels_dev_raw, dev_labels, tokens_rels_test_raw, \
                   test_labels, torch.tensor(gold_labels), torch.tensor(mislabeled), no_relation_index, noise_rate

    else:
        logger.info(f"Other class id for labeling: {no_relation_index}")
        if return_raw is False:
            return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, torch.tensor(
                gold_labels), mislabeled, no_relation_index, noise_rate
        else:
            return tokens_rels_train_raw, train_labels_ds, tokens_rels_dev_raw, dev_labels, torch.tensor(
                gold_labels), mislabeled, no_relation_index, noise_rate


def process_knodle_data_new_from_BERT(
        path_to_train: str, path_to_train_z, path_to_train_t, path_to_dev: str, storing_loc_embeddings: str,
        path_to_test: str = None, no_relation_index: int = None
):
    df_train = pd.read_csv(path_to_train, sep='\t')
    tokens_train = pd.DataFrame(df_train["sample"])
    gold_labels = df_train["gold_label"]

    z = joblib.load(path_to_train_z)
    t = joblib.load(path_to_train_t)

    noisy_y_train_probs = z_t_matrices_to_majority_vote_probs(z, t)
    zeros = np.where(noisy_y_train_probs.sum(axis=1) == 0)[0]
    non_zeros = np.where(noisy_y_train_probs.sum(axis=1) != 0)[0]
    logger.info(f"Samples without matches: {len(zeros)} out of {len(noisy_y_train_probs)}")

    # apply majority voting
    noisy_y_train = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=noisy_y_train_probs)

    # load train embeddings
    train_sentences_ds = pd.read_csv(os.path.join(storing_loc_embeddings, "train", "embeddings.csv")).to_numpy()

    # filter out samples without matches (if wanted)
    if no_relation_index is None:
        tokens_train, noisy_y_train, z, gold_labels = filter_empty_probabilities(tokens_train, noisy_y_train, zeros,
                                                                                 non_zeros, gold_labels, z)
        train_sentences_ds = train_sentences_ds[non_zeros]

    # otherwise replace them with no_relation_index
    else:
        noisy_y_train[zeros] = no_relation_index

    # prepare the labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(noisy_y_train)
    train_labels_ds = label_encoder.transform(noisy_y_train)
    gold_labels = label_encoder.transform(gold_labels)

    mislabeled = np.zeros(train_sentences_ds.shape[0])
    mislabeled[np.where(train_labels_ds != gold_labels)] = 1

    noise_rate = sum(mislabeled) / len(mislabeled)
    logger.info(f"Noise rate: {noise_rate}")

    # prepare dev set
    tokens_rels_dev = pd.read_csv(path_to_dev, sep='\t')
    dev_labels = label_encoder.transform(tokens_rels_dev["label"])

    # load dev embeddings
    dev_sentences = pd.read_csv(os.path.join(storing_loc_embeddings, "dev", "embeddings.csv")).to_numpy()

    if path_to_test is not None:
        tokens_rels_test = pd.read_csv(path_to_test, sep='\t')
        test_labels = label_encoder.transform(tokens_rels_test["label"])

        test_sentences = pd.read_csv(os.path.join(storing_loc_embeddings, "test", "embeddings.csv")).to_numpy()

        logger.info(f"Other class id for labeling: {no_relation_index}")
        return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, \
               torch.tensor(gold_labels), torch.tensor(mislabeled), no_relation_index, noise_rate

    else:
        logger.info(f"Other class id for labeling: {no_relation_index}")
        return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, torch.tensor(gold_labels), \
               torch.tensor(mislabeled), no_relation_index, noise_rate


def vectorize_data(
        tokens_rels: List[Tuple],
        vectorizer: TfidfVectorizer = None,
        default_vectorizer: str = "tfidf",
        label_encoder: LabelEncoder = None,
        max_features=1000
) -> Tuple[sp.csr_matrix, np.ndarray, CountVectorizer, LabelEncoder]:
    # list of instances
    texts = [" ".join(tr[0]) for tr in tokens_rels]

    # list of labels
    labels = [tr[1] for tr in tokens_rels]

    # if no vectorizer defined yet, fit one of the type default_vectorizer to the data
    if not vectorizer:
        if default_vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(max_features=max_features)
        elif default_vectorizer == "count":
            vectorizer = CountVectorizer(max_features=max_features)
        text_matrix = vectorizer.fit_transform(texts)

    # otherwise use provided vectorizer
    else:
        text_matrix = vectorizer.transform(texts)

    # if no label encoder provided fit one to the labels
    if not label_encoder:
        label_encoder = preprocessing.LabelEncoder()
        label_matrix = label_encoder.fit_transform(labels)
    else:
        label_matrix = label_encoder.transform(labels)

    return text_matrix, label_matrix, vectorizer, label_encoder


def z_t_matrices_to_majority_vote_probs(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray
) -> np.ndarray:
    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError(f"Dimensions mismatch! Z matrix has shape {rule_matches_z.shape}, while "
                         f"T matrix has shape {mapping_rules_labels_t.shape}")

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_counts = rule_matches_z.dot(mapping_rules_labels_t)
        if isinstance(rule_counts, sp.csr_matrix):
            rule_counts = rule_counts.toarray()
    else:
        rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)

    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)
    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def probabilies_to_majority_vote(probs: np.ndarray, filter_ties: bool = False) -> int:
    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:  # if max probability unique
        return int(np.argmax(probs))
    else:
        max_ids = np.where(probs == row_max)[0]
        if filter_ties is False:
            return int(np.random.choice(max_ids))  # choose randomly if two labels share the highest probability
        else:
            return -100


def filter_empty_probabilities(  # not used/commented out
        input_data_x: TensorDataset, noisy_y_train: np.ndarray, zeros, non_zeros, gold_labels: np.ndarray,
        rule_matches_z: np.ndarray = None
) -> Union[Tuple[TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray]]:
    new_x = input_data_x.drop(zeros)  # filter out no matches

    assert len(new_x) == noisy_y_train[non_zeros].shape[0] == rule_matches_z[non_zeros, :].shape[0]

    if rule_matches_z is not None:
        return new_x, noisy_y_train[non_zeros], rule_matches_z[non_zeros], gold_labels[non_zeros]

    return new_x, noisy_y_train[non_zeros], gold_labels[non_zeros]


def filter_tensor_dataset_by_indices(dataset: TensorDataset,
                                     filter_ids: Union[np.ndarray, List[int]]) -> TensorDataset:  # also not used
    """ COPIED FROM KNODLE, 03.12.2021
    Filters each tensor of a TensorDataset, given some "filter_ids".
    Args:
        dataset: TensorDataset with a list of tensors, each having first dimension N
        filter_ids: A list of K indices to be kept, K <= N
    Returns: TensorDataset with filtered indices
    """
    new_tensors = []
    for i in range(len(dataset.tensors)):
        new_tensors.append(dataset.tensors[i][filter_ids])
    dataset_new = TensorDataset(*new_tensors)

    return dataset_new


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
        return self.features[index], self.labels[index].long()

    def __len__(self):
        return self.len


class AGRADatasetWithGold(Dataset):
    def __init__(self, features, labels, gold_labels, mislabeled):
        self.len = labels.shape[0]

        if isinstance(features, np.ndarray) is False:

            self.features = torch.tensor(features.toarray()).float()

        else:
            self.features = torch.tensor(features).float()

        self.labels = torch.tensor(labels)
        self.gold_labels = gold_labels
        self.mislabeled = mislabeled

    def __getitem__(self, index):
        return self.features[index], self.labels[index].long(), self.gold_labels[index], self.mislabeled[index]

    def __len__(self):
        return self.len


def prepare_data(storing_loc: str, storing_loc_embeddings=None, default_label=0, seed=0, bert=False, filter_ties=False,
                 return_raw=False, default_vectorizer="tfidf"):
    np.random.seed(seed)
    if bert is False:
        train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, gold_labels, \
        label_errors, no_relation_index, noise_rate = \
            process_knodle_data_new(
                os.path.join(storing_loc, "train_df.csv"),
                os.path.join(storing_loc, "train_rule_matches_z.lib"),
                os.path.join(storing_loc, "mapping_rules_labels_t.lib"),
                os.path.join(storing_loc, "dev_df.csv"),
                os.path.join(storing_loc, "test_df.csv"),
                no_relation_index=default_label,
                filter_ties=filter_ties,
                return_raw=return_raw,
                default_vectorizer=default_vectorizer
            )

        weights = compute_weights(train_labels_ds)

        return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, \
               gold_labels, label_errors, no_relation_index, weights, noise_rate

    elif bert is True:
        train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, gold_labels,\
        label_errors, no_relation_index, noise_rate = \
            process_knodle_data_new_from_BERT(
                os.path.join(storing_loc, "train_df.csv"),
                os.path.join(storing_loc, "train_rule_matches_z.lib"),
                os.path.join(storing_loc, "mapping_rules_labels_t.lib"),
                os.path.join(storing_loc, "dev_df.csv"),
                storing_loc_embeddings,
                os.path.join(storing_loc, "test_df.csv"),
                no_relation_index=default_label
            )

        weights = compute_weights(train_labels_ds)

        return train_sentences_ds, train_labels_ds, dev_sentences, dev_labels, test_sentences, test_labels, \
               gold_labels, label_errors, no_relation_index, weights, noise_rate


def compute_weights(train_labels):
    # determine the number of classes and samples
    num_classes = len(np.unique(train_labels))
    num_samples = len(train_labels)

    # count occurrences per label
    _, counts = np.unique(train_labels, return_counts=True)

    # initialize weights, source/cite
    weights = np.zeros(num_samples)
    for label in range(0, num_classes):
        # weights[train_labels == label] = num_samples / (num_classes * counts[label])
        weights[train_labels == label] = 1 / counts[label]

    return weights


def load_model(model: str):
    if model == "roberta":
        return "roberta-base"
    elif model == "bert":
        return "bert-base-cased"
    elif model == "multilingual":
        return "bert-base-multilingual-cased"
    else:
        raise ValueError("Check the model name")
