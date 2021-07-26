# Generate Image Dataset for PyTorch
# Danni Chen
# 07/20/2021
# Modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Modified from Felipe's Dataset

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os, re
import numpy as np
import pandas as pd
from torchvision.io import read_image

## Before inputting the metadata_df, the training or testing is seperated. ##
class DanniDataset(Dataset):
    def __init__(self, metadata_df, image_dir, transform = None):

        ## Check if image_dir Exists ##
        if( not os.path.isdir(image_dir) ):
            exit('ImageDataset: The directory of the Image Does NOT Exist')
        
        self.metadata_df = metadata_df
        self.image_dir = image_dir
            
    def __len__(self):
        ## Number of rows of imageData dataframe ##
        return len(self.metadata_df)

    def __getitem__(self, idx):
        ## imageData Row:idx ## 
        image_row = self.metadata_df.iloc[idx]
        
        ## Image label ##
        label = image_row['Label']
        if label == '0':
            label = 'paper'
        else:
            label = 'metal'
            
        ## Image name(relative_path) ##
        relative_path = image_row['Relative_path']
        
        image_path = os.path.join(self.image_dir, str(relative_path))
        image = read_image(image_path)
        ## see what read_image returns (if it is a tensor)
        image = self.toTensor(image)
        
        item = {'image': image, 'label': label}
    
        return item
