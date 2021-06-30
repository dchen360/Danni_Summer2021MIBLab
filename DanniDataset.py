# Generate Image Dataset for PyTorch
# Danni Chen
# 6/27/2021
# Modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Modified from Felipe's Dataset

# import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from skimage import io
# import PIL.Image
import os, re
import numpy as np
import pandas as pd
## QC Metrics ##
# from .FocusMetric import FocusMetric
# from .GreyMetric import GreyMetric
from torchvision.io import read_image

class DanniDataset(Dataset):
    def __init__(self, imageData_df, image_dir, transform = None):

        ## Check if image_dir Exists ##
        if( not os.path.isdir(image_dir) ):
            exit('ImageDataset: The directory of the Image Does NOT Exist')
        
        self.imageData_df = imageData_df
        self.image_dir = image_dir
        self.transform = transform
        
  
        ## Each dataframe stores location of the image relative to image directory and 
        ## Image directory: C:\Users\danni\OneDrive\Mini Project\Garbage Classification Dataset\MetalPaper_Test
        ## Relative path: metal\metal1.png
        ## For relative path, can remove anything in front of metal, relative path is the speficif image from the point of view of the image directory
        
        ## need to create a dataframe and save it as CSV, do it with pandas
        ## dataframe has relative path and the image label (0 or 1)
        ## 
        
        ## Load metadata (path_to_csv or dataframe) ##
#         if (type(imageDataFile) == pd.core.frame.DataFrame):
#             imageData_df = imageDataFile.copy()
#         else:
#             imageData_df = pd.read_csv(imageDataFile)
     
            
    def __len__(self):
        ## Number of rows of imageData dataframe ##
        return len(self.imageData_df)

    def __getitem__(self, idx):
        ## imageData Row:idx ## 
#         image_row = self.imageData.iloc[idx]
#         ## Image Label (path) ##
#         label = image_row['Label'] 

        ## Image name and location/path ##
        image_path = os.path.join(self.root, str(label), image_name )
        image = read_image(image_path)
        ## see what read_image returns (if it is a tensor)
        image = self.toTensor(image)
        
        item = {'image': image, 'label': label}
    
        return item
