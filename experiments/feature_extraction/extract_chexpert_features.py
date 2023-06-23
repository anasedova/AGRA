"""
This file extracts embeddings that cab be used for experiemnts with CheXpert.
"""

## Imports

import copy
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from utils import CheXpertDatasetProcessor, sample_held_out, extract_ids, train_and_validate_model
from utils import set_seed, get_image_embedding

train_model = False

## Define storing location for model and emebeddings
file_path = os.path.split(os.path.abspath(__file__))[0]
storing_location_model = os.path.join(file_path, 'model_new')
os.makedirs(storing_location_model, exist_ok=True)

storing_location_embeddings = os.path.join(file_path, 'datasets', "chexpert", "encoded_efficientnetb0_new")
os.makedirs(storing_location_embeddings, exist_ok=True)
storing_location_train_embeddings = os.path.join(storing_location_embeddings, "train_embeddings_efficientnetb0.data")
storing_location_valid_embeddings = os.path.join(storing_location_embeddings, "valid_embeddings_efficientnetb0.data")
storing_location_test_embeddings = os.path.join(storing_location_embeddings, "test_embeddings_efficientnetb0.data")

storing_location_train_labels = os.path.join(storing_location_embeddings, "train_labels.data")
storing_location_valid_labels = os.path.join(storing_location_embeddings, "valid_labels.data")
storing_location_test_labels = os.path.join(storing_location_embeddings, "test_labels.data")

## Load and prep data
set_seed(0) # set seed for reproducibility


dataset_path = os.path.join(file_path, "CheXpert-v1.0-small")
training_set = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
validation_set = pd.read_csv(os.path.join(dataset_path, 'valid.csv'))

# define the transformations that should be applied to the images
transform_list = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization from ImageNet
])

# paths to training images
image_paths_train = [os.path.join(file_path, p) for p in training_set["Path"]]

# paths to validation images
image_paths_valid = [os.path.join(file_path,  p) for p in validation_set["Path"]]

# first 12 labels correspond to pathologies
all_pathologies = list(training_set.iloc[:, -13:-1].columns)

chexpert_train = CheXpertDatasetProcessor(path=dataset_path,
                                          subset="train",
                                          image_paths=image_paths_train,
                                          transform_sequence=transform_list,
                                          replacement_for_blank=0,
                                          to_ones=all_pathologies,  # how to handle uncertain labels
                                          only_frontal=False) # only use frontal images or not
train_set = chexpert_train.process_chexpert_dataset()
print("Size of training set:", chexpert_train.__len__(), "\n")

# prepare validation data -> called chexpert_test since it will be used as test data later on
chexpert_test = CheXpertDatasetProcessor(path=dataset_path,
                                         subset="valid",
                                         image_paths=image_paths_valid,
                                         transform_sequence=transform_list,
                                         replacement_for_blank=0,
                                         to_ones=all_pathologies,
                                         only_frontal=False)
test_set = chexpert_test.process_chexpert_dataset()
print("Size of test set:", chexpert_test.__len__(), "\n")

## Create embeddings

### Take a subset of the training data for creating a validation set
# number of patients for train and test set
num_patients = len(np.unique(extract_ids(train_set["Path"])))
print("Size of training set:", chexpert_train.__len__())
print("Number of patients in train set:", num_patients, "\n")

num_patients_test = len(np.unique(extract_ids(test_set["Path"])))
print("Size of test set:", chexpert_test.__len__())
print("Number of patients in test set:", num_patients_test, "\n")

# split off a held-out set for validation from the weak training set
chexpert_train_subset, chexpert_held_out, train_sample, held_out_sample = sample_held_out(train_set, chexpert_train, 7000)

print("Number of observations in the final training set:", chexpert_train_subset.__len__())
print("Number of observations in the final validation set:", chexpert_held_out.__len__())
print("Number of observations in the final test set:", chexpert_test.__len__(), "\n")

# just to check
print("Number of patients in the final training set:", len(np.unique(extract_ids(train_sample["Path"]))))
print("Number of patients in the final validation set:", len(np.unique(extract_ids(held_out_sample["Path"]))), "\n")

# label distribution
print("Label dist. in train set:")
print(train_sample.iloc[:, -13:-1].sum(axis=0), "\n")
print("Label dist. in dev set:")
print(held_out_sample.iloc[:, -13:-1].sum(axis=0), "\n")
print("Label dist. in test set:")
print(test_set.iloc[:, -13:-1].sum(axis=0), "\n")

# save subsets
test_set_filtered = copy.deepcopy(test_set)
if train_model is True:
    torch.save(held_out_sample.index, os.path.join(storing_location_model, "validation_indices" + ".pt"))
print(torch.load(os.path.join(storing_location_model, "validation_indices" + ".pt")))
print(held_out_sample.index)
train_sample.drop(["No Finding", "Support Devices"], inplace=True, axis=1)
held_out_sample.drop(["No Finding", "Support Devices"], inplace=True, axis=1)
test_set_filtered.drop(["No Finding", "Support Devices"], inplace=True, axis=1)

# reset indices
train_sample = train_sample.reset_index(drop=True)
held_out_sample = held_out_sample.reset_index(drop=True)

if train_model is True:
    train_sample.to_csv(os.path.join(dataset_path, 'train_filtered.csv'))
    held_out_sample.to_csv(os.path.join(dataset_path, 'held_out_filtered.csv'))
    test_set_filtered.to_csv(os.path.join(dataset_path, 'test_filtered.csv'))

# number of classes
num_classes = len(chexpert_train_subset.__getitem__(0)[1])
print("Number of classes:", num_classes)

### Fine-tune Efficientnet-B0
# create data loaders
batch_size = 16
train_loader = DataLoader(chexpert_train_subset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(chexpert_held_out, batch_size=batch_size)
test_loader = DataLoader(chexpert_test, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if train_model is True:
    # load Efficientnet-b0 from torchvision
    model = models.efficientnet_b0(pretrained=True)
    embedding_size = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(embedding_size, num_classes)

    # set parameters for training (like in CheXpert paper)
    num_epochs = 2
    batch_size = 16
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train the model for two epochs
    train_and_validate_model(model, num_epochs, train_loader, valid_loader, criterion, optimizer, device, storing_location_model)

# load the weights
final_model = models.efficientnet_b0()
checkpoint = torch.load(os.path.join(storing_location_model, "efficientnet_epoch_2" + ".pt"))
embedding_size = final_model.classifier[1].in_features
final_model.classifier[1] = nn.Linear(embedding_size, num_classes)
final_model.load_state_dict(checkpoint['model_state_dict'])
final_model = final_model.to(device)
final_model.eval()

probas = []
y_true = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
            data, target = data[0].to(device), data[1].to(device)
            outputs = torch.sigmoid(final_model(data))
            probas.append(outputs.cpu().detach().numpy())
            y_true.append(target.cpu())

probas = np.vstack(probas)
y_true = np.vstack(y_true)
valid_metric = roc_auc_score(y_true, probas, average='macro')
print('Valid performance of extraction model:', valid_metric)

### Extract embeddings
train_loader_embeddings = DataLoader(chexpert_train_subset, batch_size=batch_size, shuffle=False)
train_embeddings = get_image_embedding(train_loader_embeddings, device='cuda:0', model=final_model)
joblib.dump(train_embeddings, storing_location_train_embeddings)
valid_embeddings = get_image_embedding(valid_loader, device='cuda:0', model=final_model)
joblib.dump(valid_embeddings, storing_location_valid_embeddings)
test_embeddings = get_image_embedding(test_loader, device='cuda:0', model=final_model)
joblib.dump(test_embeddings, storing_location_test_embeddings)

# save labels
train_labels = train_sample.iloc[:, -12:].values
joblib.dump(train_labels, storing_location_train_labels)
valid_labels = held_out_sample.iloc[:, -12:].values
joblib.dump(valid_labels, storing_location_valid_labels)
test_labels = test_set_filtered.iloc[:, -12:].values
joblib.dump(test_labels, storing_location_test_labels)