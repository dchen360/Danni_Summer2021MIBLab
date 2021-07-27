import numpy as np
import pandas as pd
import os, re
        
## Each dataframe stores location of the image relative to image directory, the label of the image, and whether the image is for training or testing
## image_dir: r'C:\Users\danni\OneDrive\Mini Project\Garbage Classification Dataset'
        
## To dos:
## ! image_dir should be able to see all images ## check: crop the image_dir off to before splitting the groups
## one more col in metadata_df, if training ## check
## !! Be in another separate jupyter notebook named Create_Metadata, saving as a csv
        
## Relative path: metal\metal1.png
## Image label (0:paper and 1:metal)
## For relative path, can remove anything in front of metal, relative path is the speficif image from the point of view of the image directory
        
## Creating a dictionary that has two keys: 'Relative_path' and 'Label' with two lists being the values
        
data_info = {}
rel_file_list = []
label_list = []
if_train = []

def create_metadata(image_dir):
        for dir_, _, files in os.walk(image_dir):
                    for file_name in files:
                                rel_dir = os.path.relpath(dir_, image_dir)
                                train = rel_dir[11:16]
                                if train == 'Train':
                                        if_train.append(1)
                                else:
                                        if_train.append(0)
        
                                rel_file = os.path.join(rel_dir, file_name)
                                rel_file_list.append(rel_file)
                        
                                alist = rel_file.split('\\')
                                label = alist[1]
                                if label == 'paper':
                                        label = '0'
                                elif label == 'metal':
                                        label = '1'
                                label_list.append(label)
                                data_info['Relative_path'] = rel_file_list
                                data_info['Label'] = label_list
                                data_info['Train'] = if_train

        metadata_df = pd.DataFrame(data_info)
        metadata_df.to_csv(image_dir + '\metadata.csv')
        return metadata_df
