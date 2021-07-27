import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from DanniDataset import DanniDataset

## User_defined variables (anything i need to type out/hyperparameters): image dir, the location of the csv file, batch size, learning rate

## What to put for the parameters? The image directory, the metadata_df, and wether it transform (an object)? ##
## train_dataset and test_dataset ##
train_dataset = DanniDataset('train_metadata.csv', 'C:\Users\danni\OneDrive\Mini Project\Garbage Classification Dataset')
test_dataset =  DanniDataset('test_metadata.csv', 'C:\Users\danni\OneDrive\Mini Project\Garbage Classification Dataset')

## Customized dataset is for retrieving the data using data loader ##
## For more information of DataLoader refer to: https://pytorch.org/docs/stable/data.html ##
## The image_dataset would be a map-style dataset. 
## Such a dataset, implements the __getitem__() and __len__() protocols. ##
## When accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk. ##

## train_loader and test_loader ##
## test_loader, shuffle = False ##
## For training, the order of the images that the model see should be random, or the model can memorize the pattern. ##
## For testing, the order does not matter because the parameter is not being updated. ##
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=1
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    num_workers=1
)

# To check if the DanniDataset works properly. ##
## Get the 5th image from the test_dataloader ##
idx = 5
test_image, class_label = train_loader.__getitem__(idx)
plt.figure()
plt.imshow(test_image)
plt.title(f'Image at index {idx}. Class Label {class_label}')
plt.show()

## ---------------------------------------------------------------------------------------------------------------------------------##

## Start constructing the covolution layer as image_loader works properly ##
## Build the CNN classification model. ##

## Create another jupyter notebook following the instruction of https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html ##

