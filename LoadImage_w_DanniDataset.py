import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from DanniDataset import DanniDataset

## What to put for the parameters? ##
image_dataset = DanniDataset(parameters)

## Customized dataset is for retrieving the data using data loader ##
## For more information of DataLoader refer to: https://pytorch.org/docs/stable/data.html ##
## The image_dataset would be a map-style dataset. 
## Such a dataset, implements the __getitem__() and __len__() protocols. ##
## When accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk. ##

## Do I need to get different image_dataset for train_image_loader and test_image_loader? ##
image_loader = torch.utils.data.DataLoader(
    image_dataset, batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=1
)

# To check if the DanniDataset works properly. ##
## Get the 5th image from the test_dataloader ##
idx = 5
test_image, class_label = image_loader.__getitem__(idx)
plt.figure()
plt.imshow(test_image)
plt.title(f'Image at index {idx}. Class Label {class_label}')
plt.show()

## Start constructing the covolution layer as image_loader works properly ##

