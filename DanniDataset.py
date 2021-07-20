# Generate Image Dataset for PyTorch
# Danni Chen
# 07/20/2021
# Modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Modified from Felipe's Dataset

# import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os, re
import numpy as np
import pandas as pd
from torchvision.io import read_image

class DanniDataset(Dataset):
    def __init__(self, imageData_df, image_dir, transform = None):

        ## Check if image_dir Exists ##
        if( not os.path.isdir(image_dir) ):
            exit('ImageDataset: The directory of the Image Does NOT Exist')
        
        self.imageData_df = imageData_df
        self.image_dir = image_dir
        
        ## Each dataframe stores location of the image relative to image directory and the label of the image
        ## image_dir: r'C:\Users\danni\OneDrive\Mini Project\Garbage Classification Dataset\MetalPaper_Train'
        ## Relative path: metal\metal1.png
        ## Image label (0:paper and 1:metal)
        ## For relative path, can remove anything in front of metal, relative path is the speficif image from the point of view of the image directory
        
        ## Creating a dictionary that has two keys: 'Relative_path' and 'Label' with two lists being the values
        data_info = {}
        rel_file_list = []
        label_list = []

        for dir_, _, files in os.walk(self.image_dir):
            for file_name in files:
                rel_dir = os.path.relpath(dir_, root_dir)
                rel_file = os.path.join(rel_dir, file_name)
                rel_file_list.append(rel_file)
                label = rel_file[0:5]
                if label == 'paper':
                    label = '0'
                else:
                    label = '1'
                label_list.append(label)
         data_info['Relative_path'] = rel_file_list
         data_info['Label'] = label_list

         imageData_df = pd.DataFrame(data_info)
            
    def __len__(self):
        ## Number of rows of imageData dataframe ##
        return len(self.imageData_df)

    def __getitem__(self, idx):
        ## imageData Row:idx ## 
        image_row = self.imageData_df.iloc[idx]
        ## Image label ##
        label = image_row['Label']
        ## Image name(relative_path) ##
        relative_path = image_row['Relative_path']
        
        image_path = os.path.join(self.image_dir, str(relative_path))
        image = read_image(image_path)
        ## see what read_image returns (if it is a tensor)
        image = self.toTensor(image)
        
        item = {'image': image, 'label': label}
    
        return item
