#Pytorch ImageDataset

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import random
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

class Train_val_ImageDataset(Dataset):
    def __init__(self, data_file, state, image_columns, target_columns=None, delete_columns=None, transform=None):
        self.state = state
        if self.state == 'train':
            self.data = pd.read_csv(data_file)
        else:
            self.data = pd.read_csv(data_file)
        self.image_columns = image_columns
        self.target_columns = target_columns if target_columns is not None else []
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.transform = transform

    def __getitem__(self, idx):
        if len(self.target_columns) == 0:
            image = Image.open(self.data[self.image_columns][idx])
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            image = Image.open(self.data[self.image_columns][idx])
            if self.transform is not None:
                image = self.transform(image)
            return image, self.data[self.target_columns][idx]

    def __len__(self):
        return len(self.data)


    #Pytorch TextDataset
    import torch