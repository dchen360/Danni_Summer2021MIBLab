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
    def __init__(self, imageDataFile, image_dir, transform = None):

        ## Check if image_dir Exists ##
        if( not os.path.isdir(image_dir) ):
            exit('ImageDataset: The directory of the Image Does NOT Exist')
        
        self.image_dir = image_dir
        self.imageData = imageDataFile
        self.transform = transform
        
        ## Load metadata (path_to_csv or dataframe) ##
#         if (type(imageData) == pd.core.frame.DataFrame):
#             imageData_df = imageData.copy()
#         else:
#             imageData_df = pd.read_csv(imageData)
            
    def __len__(self):
        ## Number of rows of imageData dataframe ##
        return len(self.imageDataFile)

    def __getitem__(self, idx):
        ## imageData Row:idx ## 
#         image_row = self.imageData.iloc[idx]
#         ## Image Label (path) ##
#         label = image_row['Label'] 
#         ## WSI Name ##
#         ID = image_row['ID']
        ## Image name and location/path ##
#         image_name = image_row['Patch_name']
#         image_path = os.path.join(self.root, str(label), image_name )
#         image = self.toTensor(image)

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        item = {'image': image, 'label': label}
    
        return item

        ## Print Original Shape ##
#        print('Metadata Shape: %s'% (metadata_df.shape,) )
        ## Number of WSI Represented ##
#        print('Number of WSIs (Original): %s'%(np.unique( np.array(metadata_df['ID']) ).shape,) )

        ## Ignore QC Failed Patches ##
#         failed_Focus= metadata_df['F_Focus'] < focus_threshold
#         failed_Grey = metadata_df['F_Grey']  > grey_threshold
#         ignore_rows = (failed_Grey | failed_Focus)
#         metadata_df.loc[ignore_rows, 'Label'] = 'Ignore'
#         ## Remove Label = 'Ignore' ##
#         metadata_df = metadata_df[metadata_df['Label'] != 'Ignore']
#         print('Filtered Shape (QC): %s'% (metadata_df.shape,) )

        ## Ignore WSI with less than nTilesThresh tiles ##
#         WSI_nTiles= metadata_df.groupby('ID')['Patch_name'].count()
#         ## List of WSIs not meeting threshold ##
#         ignore_list = WSI_nTiles < nTilesThresh
#         ignore_list= list( ignore_list[ ignore_list ].index )
#         ignore_rows = [ WSI_id in ignore_list for WSI_id in metadata_df['ID'] ] 
#         metadata_df.loc[ignore_rows, 'Label'] = 'Ignore'
#         ## Remove Label = 'Ignore' ##
#         metadata_df = metadata_df[metadata_df['Label'] != 'Ignore']
#         print('Filtered Shape (nTilesThresh): %s'% (metadata_df.shape,) )
#         ## Number of WSI Represented ##
#         print('Number of WSIs (Final): %s'%(np.unique( np.array(metadata_df['ID']) ).shape,) )
        
        
        ## Color Augmentation ##
#         self.augment_color = transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=1., hue=0.5) # V18: Superset
#         self.augment_color = transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.5, hue=0.05) # V17: Natural
#         self.augment_color = transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=1., hue=0.5) # V16: Superset
#         self.augment_color = transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.5, hue=0.05) # V13-V15: Natural
#         self.augment_color = transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.5, hue=0.0) # V10-12
#         self.augment_color = transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.3)
#         self.augment_color = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.4)
        
        ## Flip Augmentation ##
#         augment_flips = [ transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5) ]
        ## Apply list of tranformation functions ##
#         self.transforms_flips = transforms.Compose([
#             *augment_flips, # PIL
#         ])
        ## PIL to Tensor ##
#         self.toTensor = transforms.ToTensor()
        
        ## Augment Boolean ##
#         self.augment = augment
        ## Metadata attribute ##
#         self.metadata = metadata_df
        ## Image Directory ##
#         self.root = root
        ## QC Thresholds ##
#         self.focus_threshold = focus_threshold
#         self.grey_threshold = grey_threshold
        ## Augment image_2 (target) ##
#         self.augment_target = augment_target
        
        
        ## Load image into numpy array with PIL ##
#        image = PIL.Image.open(image_path)
        
#         ## Apply Transformations ##
#         if(self.augment):
#             ### Apply transforms to Image ###
#             ## Apply: Flips ##
#             image = self.transforms_flips(image)
#             ## Second Image ##
#             image_2 = image.copy()
#             ## Apply: Hue Shifts ##
#             image   = self.augment_color( image )
#             ## Augment image_2 (target) ##
#             if(self.augment_target):
#                 image_2 = self.augment_color( image_2 )
#             ## Apply: Tensor ##
#             image   = self.toTensor(image)
#             image_2 = self.toTensor(image_2)
        
#             ## Item to return ##
#             item = {'image': image, 'label': label, 'ID':ID, 'image_2':image_2}
            
#         else: # No Augment
#             ## Apply: Tensor ##
#             image = self.toTensor(image)
#             ## Item to return ##
#             item = {'image': image, 'label': label, 'ID':ID}
        

