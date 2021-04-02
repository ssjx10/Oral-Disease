import os
import torch 
import pandas as pd
import random
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class Cancer_train_dataset(Dataset):
    def __init__(self, data_list, transform, ben_prepro=False, data_dir='/DATA/data_cancer/train'):
        
        self.data_dir = data_dir
        self.transform = transform
        self.data = data_list
        self.ben_prepro = ben_prepro

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_dir, self.data[idx])
        
        if not os.path.exists(image_path):
            print('dose not exist '+image_path)

        label = self.data[idx].split('_')[-1].split('.')[0]
        if label == 'B' or label == 'N':
            label = 0
        elif label == 'C':
            label = 1

        '''image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)'''

        image = Image.open(image_path)
        
        if self.ben_prepro:
            image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']

        sample = {'image':image, 'label':label}

        return sample

class Cancer_test_dataset(Dataset):
    def __init__(self, data_list, transform, ben_prepro=False, data_dir='/DATA/data_cancer/test'):

        self.data_dir = data_dir
        self.transform = transform
        self.data = data_list
        self.ben_prepro = ben_prepro

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_dir, self.data[idx])
        image_name = self.data[idx]

        if not os.path.exists(image_path):
            print('dose not exist '+image_path)

        '''image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)'''
        image = Image.open(image_path)

        if self.ben_prepro:
            image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']


        sample = {'image':image, 'image_name': image_name}

        return sample
