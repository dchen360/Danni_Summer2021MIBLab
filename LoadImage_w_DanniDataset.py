import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from DanniDataset import DanniDataset

## What to put for the parameters? ##
image_dataset = DanniDataset(parameters)

## Do I need to get different image_dataset for train_image_loader and test_image_loader? ##
image_loader = torch.utils.data.DataLoader(
    image_dataset, batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=1
)
