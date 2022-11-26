"""
This file shows how to extract embeddings for the images in the CheXpert dataset.
"""

## Imports

import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import CheXpertDatasetProcessor, sample_held_out, filter_studies, extract_ids, train_and_validate_model

## Define storing location for model

"""
Please replace "model_path" with the path to the location where the model should be stored
"""

storing_location_model = r"C:\Users\Lena\Documents\Data Science\Masterarbeit\Chexpert\new_models"

## Load and prep data

np.random.seed(0) # set seed for reproducibility of train, val split

"""
Please replace "data_path" with the path to the location where you have train.csv and valid.csv stored
"""

path = r"C:\Users\Lena\Documents\Data Science\Semester 3\DS Project\Data\CheXpert-v1.0-small\CheXpert-v1.0-small"
os.chdir(path)  # change working directory to appropriate location

training_set = pd.read_csv('train.csv')
validation_set = pd.read_csv("valid.csv")

"""
The embeddings created in this tutorial will be saved in the same location in folders called "train_embeddings", "valid_embeddings" and "test_embeddings" respectively
"""

# define transformations
transform_list = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization from ImageNet
])

# define paths to images
split_path = os.path.split(path)[0]

# paths to training images
image_paths_train = [os.path.join(split_path, p) for p in training_set["Path"]]

# paths to validation images
image_paths_valid = [os.path.join(split_path, p) for p in validation_set["Path"]]

all_pathologies = list(training_set.iloc[:, -13:-1].columns)

chexpert_train_full = CheXpertDatasetProcessor(path=path,
                                                subset="train",
                                                image_paths=image_paths_train,
                                                transform_sequence=transform_list,
                                                replacement_for_blank=0,
                                                to_ones=all_pathologies, # how to handle uncertain labels
                                                only_frontal=True)
train_set_frontal = chexpert_train_full.process_chexpert_dataset()
print("Size of training set (only frontal images):", chexpert_train_full.__len__())

# prepare validation data -> called chexpert_test since will be used as test data later on
chexpert_test = CheXpertDatasetProcessor(path=path,
                                         subset="valid",
                                         image_paths=image_paths_valid,
                                         transform_sequence=transform_list,
                                         replacement_for_blank=0,
                                         to_ones=all_pathologies,
                                         only_frontal=True)
test_set_frontal = chexpert_test.process_chexpert_dataset()
print("Size of test set (only frontal images):", chexpert_test.__len__())

## Create embeddings

### Take a subset of the training data for creating a validation set

# remove studies with more than one view
train_filtered = filter_studies(train_set_frontal)
test_filtered = filter_studies(test_set_frontal)

# retrieve index of remaining samples and only take this subset for train dataloader
chexpert_train_filtered = torch.utils.data.Subset(chexpert_train_full, train_filtered.index)

# reset the index of the dataframe after filtering
train_filtered = train_filtered.reset_index(drop=True)
num_patients = len(np.unique(extract_ids(train_filtered["Path"])))

print("Size of training set after filtering out studies with more than one frontal view:", chexpert_train_filtered.__len__())
print("Number of patients in train set after filtering out studies with more than one frontal view:", num_patients)

# repeat the same for test set
chexpert_test_filtered = torch.utils.data.Subset(chexpert_test, test_filtered.index)
test_filtered = test_filtered.reset_index(drop=True)
print("Size of test set after filtering out studies with more than one frontal view:", chexpert_test_filtered.__len__())

# split off a held-out set for validation
chexpert_train_subset, chexpert_held_out, train_sample, held_out_sample = sample_held_out(train_filtered, chexpert_train_filtered, 7000)

# save the indices of the observations in the held_out set
torch.save(held_out_sample.index, os.path.join(storing_location_model, "validation_indices" + ".pt"))

print("Number of observations in the final training set:", chexpert_train_subset.__len__())
print("Number of observations in the final validation set:", chexpert_held_out.__len__())
print("Number of observations in the final test set:", chexpert_test_filtered.__len__())

# label distribution
print("Label dist. in train set:")
print(train_sample.iloc[:, -13:-1].sum(axis=0))
print("Label dist. in dev set:")
print(held_out_sample.iloc[:, -13:-1].sum(axis=0))
print("Label dist. in test set:")
print(test_filtered.iloc[:, -13:-1].sum(axis=0))

# number of classes
num_classes = len(chexpert_train_subset.__getitem__(0)[1])
print("Number of classes:", num_classes)

### Fine-tune EfficientNet

# load EfficientNet-B0 from torchvision
model = models.efficientnet_b0(pretrained=True)

# modify last layer of model to fit 12 classes
embedding_size = model.classifier[1].in_features
model.classifier[1] = nn.Linear(embedding_size, num_classes)

# set parameters for training
num_epochs = 2
batch_size = 16
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
criterion = nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # source: https://debuggercafe.com/multi-label-image-classification-with-pytorch-and-deep-learning/

# create data loaders
train_loader = DataLoader(chexpert_train_subset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(chexpert_held_out, batch_size=batch_size)
test_loader = DataLoader(chexpert_test_filtered, batch_size=batch_size)

# train the model
# train_and_validate_model(model, num_epochs, train_loader, valid_loader, criterion, optimizer, device, storing_location_model)

# load the model
chosen_epoch = 2
checkpoint = torch.load(os.path.join(storing_location_model, "model_epoch_" + str(chosen_epoch) + ".pt"))

# load the weights
final_model = models.efficientnet_b0(pretrained=True)
final_model.classifier[1] = nn.Linear(embedding_size, num_classes)
final_model.load_state_dict(checkpoint['model_state_dict'])

### Extract embeddings
# main source for function: https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008/6
def extract_embeddings_multilabel(model: object,
                                  embedding_size: int,
                                  dataset: Dataset,
                                  storing_location: str):
    """
    Args:
        model: fine-tuned torchvision CNN
        embedding_size: size of the embeddings (depends on chosen CNN)
        dataset:
        storing_location: path to the storing location
    Returns:
        None
    """

    # remove last layer
    modules = list(model.children())[:-1]
    model_embed = torch.nn.Sequential(*modules)
    model_embed.eval()

    # freeze parameters
    for p in model_embed.parameters():
        p.requires_grad = False

    for i, ex in enumerate(tqdm(dataset)):
        with torch.no_grad():
            embedding = model_embed(ex[0].unsqueeze(0)) # ex[0] is the image, ex[1] the label
            embedding = embedding.data # not necessary, doesn't change output
            embedding = embedding.reshape(embedding_size)

            all_labels = ex[1]

            row = np.append(embedding.numpy(), all_labels.numpy())
            with open(storing_location, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(row)
                f.close()

# storing_location_train_embeddings = os.path.join(storing_location_embeddings, "train_embeddings.csv")
# storing_location_valid_embeddings = os.path.join(storing_location_embeddings, "valid_embeddings.csv")
# storing_location_test_embeddings = os.path.join(storing_location_embeddings, "test_embeddings.csv")

# extract_embeddings_multilabel(final_model, embedding_size, chexpert_train_subset, None)
# extract_embeddings_multilabel(final_model, embedding_size, chexpert_held_out, storing_location_valid_embeddings)
# extract_embeddings_multilabel(final_model, embedding_size, chexpert_test_filtered, storing_location_test_embeddings)

