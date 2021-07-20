import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from DanniDataset import DanniDataset

## What to put for the parameters? The image directory? ##
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

## ---------------------------------------------------------------------------------------------------------------------------------##

## Start constructing the covolution layer as image_loader works properly ##
## Build the CNN classification model. ##
## Why using 'relu' as the activation function?? How to optimize the hyper-parameters for the model??

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_generator, epochs=100, steps_per_epoch=2276//32,validation_data=validation_generator, validation_steps=251//32)
