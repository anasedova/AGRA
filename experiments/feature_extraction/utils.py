import collections
import copy
import os
import random
import re
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CheXpertDatasetProcessor():

    def __init__(self,
                 path: str,
                 subset: str,
                 image_paths: List[str],
                 transform_sequence: object = None,
                 to_ones: List[str] = None,
                 to_zeros: List[str] = None,
                 replacement_for_blank: int = 0,
                 only_frontal: bool = False
                 ):

        """
        Args:
            path: path to the folder where train.csv and valid.csv are stored
            subset: "train": load train.csv, "valid": load valid.csv
            image_paths: paths to the images
            transform_sequence: sequence used to transform the images
            to_ones: list of pathologies for which uncertainty labels should be replaced by 1.0
            to_zeros: list of pathologies for which uncertainty labels should be replaced by 0.0
            replacement_for_blank: value that should be used to replace the "blank" labels
        Returns:
            224 x 224 image tensor and a corresponding tensor containing 12 labels
        """

        self.path = path
        self.subset = subset
        self.image_paths = image_paths
        self.transform_sequence = transform_sequence
        self.to_ones = to_ones
        self.to_zeros = to_zeros
        self.replacement_for_blank = replacement_for_blank
        self.only_frontal = only_frontal

    def process_chexpert_dataset(self):

        # read in dataset
        if self.subset == "train":
            data = pd.read_csv(os.path.join(self.path, "train.csv"))

        elif self.subset == "valid":
            data = pd.read_csv(os.path.join(self.path, "valid.csv"))

        else:
            raise ValueError("Invalid subset, please choose either 'train' or 'valid'")

        pathologies = data.iloc[:, -13:-1].columns

        # prepare labels
        data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"),
                                                            self.replacement_for_blank)  # blank labels -> specified value

        # set uncertainty labels to 1 for pathologies in to_ones
        if self.to_ones is not None:
            if all(p in pathologies for p in self.to_ones):  # check whether arguments are valid pathologies
                data[self.to_ones] = data[self.to_ones].replace(-1, 1)  # replace uncertainty labels with 1
            else:
                raise ValueError("List supplied to to_ones contains invalid pathology, please choose from:",
                                 list(pathologies))

        # set uncertainty labels to 0 for pathologies in to_zeros
        if self.to_zeros is not None:
            if all(p in pathologies for p in self.to_zeros):
                data[self.to_zeros] = data[self.to_zeros].replace(-1, 0)  # replace uncertainty labels with 0
            else:
                raise ValueError("List supplied to to_zeros contains invalid pathology, please choose from:",
                                 list(pathologies))

        # can select to only look at frontal images
        if self.only_frontal is True:
            data = data[data["Frontal/Lateral"] == "Frontal"]
            data = data.reset_index(drop=True)

        self.number_of_images = data.shape[0]
        self.data = data

        return data

    def __getitem__(self, index: int):

        """
        index: index of example that should be retrieved
        """

        image_labels = self.data.iloc[index, -13:-1]
        image_name = self.image_paths[index]
        patient_image = Image.open(image_name).convert('RGB')

        if self.transform_sequence is not None:
            patient_image = self.transform_sequence(patient_image)  # apply the transform_sequence if one is specified
        else:
            # even if no other transformation is applied, the image should be turned into a tensor
            to_tensor = transforms.ToTensor()
            patient_image = to_tensor(patient_image)

        return patient_image, torch.FloatTensor(image_labels)

    def __len__(self):
        return self.number_of_images


class CheXpertImageLoader():

    def __init__(self,
                 label_df: pd.DataFrame,
                 image_paths: list,
                 transform_sequence: object = None,
                 ):

        self.label_df = label_df
        self.image_paths = image_paths
        self.transform_sequence = transform_sequence
        self.number_of_images = len(image_paths)
        self.labels = torch.tensor(self.label_df.iloc[:, -12:].values)

    def __getitem__(self, index: int):

        """
        index: index of example that should be retrieved
        """

        image_labels = self.labels[index, :]  # last 12 entries correspond to labels
        image_name = self.image_paths[index]
        patient_image = Image.open(image_name).convert('RGB')

        if self.transform_sequence is not None:
            patient_image = self.transform_sequence(patient_image)  # apply the transform_sequence if one is specified

        else:
            # even if no other transformation is applied, the image should be turned into a tensor
            to_tensor = transforms.ToTensor()
            patient_image = to_tensor(patient_image)

        return {'ids': index, 'features': patient_image, 'label': image_labels}

    def __len__(self):
        return self.number_of_images


class AGRAImageDataSet(TensorDataset):
    def __init__(self, *tensors: Tensor, **kwargs):
        super(TensorDataset, self).__init__(**kwargs)
        self.tensors = tensors
        self.ids = []
        self.features = []
        self.labels = []
        self.weak_labels = []
        self.examples = []

        if len(self.tensors) > 0:
            self.add_values()

    def create_subset(self, idx: List[int]):
        dataset = self.__class__()
        for i in idx:
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])

        # for multilabel setting
        dataset.labels = torch.vstack(dataset.labels)
        dataset.weak_labels = torch.vstack(dataset.weak_labels)

        if self.features is not None:
            dataset.features = self.features[idx]

        return dataset

    def add_values(self):
        self.ids = list(range(len(self.tensors[0])))
        self.features = self.tensors[0]
        self.labels = self.tensors[1]
        self.weak_labels = self.tensors[1]  # same as self.labels
        self.examples = self.tensors[0]  # same as self.features


def extract_patient_and_study(path: object):
    """
    Args:
        paths: "Path" column of train.csv
    Returns:
        list of patient ids extracted from the paths
    """

    return ([re.findall(r'patient\d+\/study\d+', path)])


def filter_studies(df: pd.DataFrame):
    data = copy.deepcopy(df)
    paths = data["Path"]
    data["Patient Study"] = [re.findall(r'patient\d+\/study\d+', path)[0] for path in paths]

    # remove studies with more than two views
    counter = collections.Counter(data["Patient Study"])
    number_of_views_per_study = dict(counter)
    correct_studies = [s for s, v in number_of_views_per_study.items() if v == 1]  # only one frontal view per study

    data = data[data["Patient Study"].isin(correct_studies)]
    data = data.drop('Patient Study', axis=1)

    return data  # index not yet reset


def extract_ids(paths: object):
    """
    Args:
        paths: "Path" column of train.csv
    Returns:
        list of patient ids extracted from the paths
    """

    return ([re.findall(r'patient\d+', path)[0] for path in paths])


def sample_held_out(data: pd.DataFrame, dataset: Dataset, number_of_patients_valid: int, seed: int = 0):
    """
    Args:
        data: dataframe to take sample from
        dataset: the corresponding dataset
        number_of_patients_valid: number of patients that should be assigned to the validation set
        seed: seed used for sampling
    Returns:
        chexpert_train_subset: a subset of the chexpert_train_filtered dataset used for training
        chexpert_held_out: a subset of the chexpert_train_filtered dataset used as held out
        train_sample: the rows in train.csv corresponding to chexpert_train_subset
        held_out_sample: the rows in train.csv corresponding to chexpert_held_out
    """

    set_seed(seed)
    train_set = copy.deepcopy(data)

    # add a column with patient id to the training df
    ids = extract_ids(data["Path"])
    train_set.insert(1, "Id", ids)

    # select patients for validation set randomly, but patients should not overlap with those in the training set
    held_out_idx = np.random.choice(np.unique(ids), size=number_of_patients_valid, replace=False)

    train_sample = train_set[~train_set.Id.isin(held_out_idx)]
    held_out_sample = train_set[train_set.Id.isin(held_out_idx)]

    # remove Id column again
    train_sample = train_sample.drop('Id', axis=1)
    held_out_sample = held_out_sample.drop('Id', axis=1)

    # create subsets from the datasets
    train_idx = train_sample.index
    held_out_idx = held_out_sample.index
    chexpert_train_subset = torch.utils.data.Subset(dataset, train_idx)
    chexpert_held_out = torch.utils.data.Subset(dataset, held_out_idx)

    return chexpert_train_subset, chexpert_held_out, train_sample, held_out_sample


# source: https://debuggercafe.com/multi-label-image-classification-with-pytorch-and-deep-learning/
def train_and_validate_model(model,
                             number_of_epochs: int,
                             dataloader_train,
                             dataloader_valid,
                             criterion: object,
                             optimizer,
                             device: str,
                             storing_location_model: str):
    """
    Args:
        model: CNN architecture that we want to fine-tune
        number_of_epochs: number of epochs the model should be trained for
        dataloader_train: dataloader for the training data
        dataloader_valid: dataloader for the validation data
        num_obs_train: number of observations in the training set
        num_obs_valid: number of observations in the validation set
        criterion: criterion that should be used for optimization
        optimizer: optimizer that is used during training
        device: either GPU or CPU
        storing_location_model: path to the folder where the resulting parameters of the CNN should be stored
    Returns:
        None
    """

    start_time = time.time()

    for epoch in range(0, number_of_epochs):
        print("Starting epoch", str(epoch))
        train_running_loss = 0.0
        valid_running_loss = 0.0
        counter_train = 0
        counter_valid = 0

        # train
        model.to(device)
        model.train()
        for i, data in enumerate(tqdm(dataloader_train)):
            counter_train += 1
            data, target = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(data))
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        # validate
        model.eval()
        probas = []
        y_true = []
        with torch.no_grad():
            for i, data in enumerate(dataloader_valid):
                counter_valid += 1
                data, target = data[0].to(device), data[1].to(device)
                outputs = torch.sigmoid(model(data))
                probas.append(outputs.cpu().detach().numpy())
                y_true.append(target.cpu())
                loss = criterion(outputs, target)
                valid_running_loss += loss.item()

        # compute performance
        train_loss = train_running_loss / counter_train
        valid_loss = valid_running_loss / counter_valid

        print("Training loss in epoch", str(epoch + 1) + ":", train_loss)
        print("Validation loss in epoch", str(epoch + 1) + ":", valid_loss)

        torch.save({'model_state_dict': model.state_dict()},
                   os.path.join(storing_location_model, 'efficientnet_epoch_' + str(epoch + 1) + ".pt"))

    end_time = time.time()
    print("Fine-tuning took:", end_time - start_time, "seconds")


def get_image_embedding(train_loader: DataLoader, encoding: str = "resnet50", device: str = 'cpu',
                        model=None) -> np.ndarray:
    """
    extract image embeddings
    https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/
    https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    """

    out_embeddings = []

    def copy_embedding(m, i, o):
        o = o[:, :, 0, 0].detach().cpu().numpy()
        out_embeddings.append(o)

    if model is None and encoding == "resnet50":
        model = models.resnet50(pretrained=True)
        model = model.to(device)
    else:
        model = model.to(device)

    layer = model._modules.get('avgpool')
    layer.register_forward_hook(copy_embedding)

    model.eval()
    for X, y in tqdm(train_loader):
        model(X.to(device))
    embeddings = [item for sublist in out_embeddings for item in sublist]
    embeddings = np.stack(embeddings, axis=0)

    return embeddings