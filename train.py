import torch
import os
import random
import numpy as np
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2, ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from src.Cancer_dataset import Cancer_train_dataset, Cancer_test_dataset
from torchvision import models
import torch.nn as nn
from src.utils import *
import torchvision
import time

from src.classifier_utils import train,test, submit, dummy

seed=20
seed_everything(seed)

n_cls=2
ben_prepro = False
image_size = 456
batch_size = 8
n_epochs = 16
lr = 1e-6
Full_setting= True
# lr = 2.5e-5
 
dummy()

#scale_te = A.Resize(256, 256)
scale = A.Resize(image_size, image_size)
#rcrop = A.RandomCrop(width=256, height=256)
rotate = A.Rotate(limit=15)
H_flip = A.HorizontalFlip(p=0.5)
V_flip = A.VerticalFlip(p=0.5)
cutout = A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5)
elastic = A.ElasticTransform(p=0.5)
RBC = A.RandomBrightnessContrast(p=0.5)
blur = A.Blur(blur_limit=24, p=0.5)
HSV = A.HueSaturationValue(p=0.5)
ssr = A.ShiftScaleRotate(rotate_limit=30, p=0.5)

normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
to_tensor = ToTensor()

train_transform = A.Compose(
        [
            scale,
            #rcrop,
            #rotate,
            H_flip,          
            V_flip,
            ssr,
            elastic,
            #blur,
            cutout,
            #RBC,

            normalize,
            to_tensor
        ])

test_transform = A.Compose(
        [
            scale,
            #scale_te,
            normalize,
            to_tensor
        ])


train_list, val_list = train_val_split('/DATA/data_cancer/train', train_ratio=0.8)
test_list = os.listdir('/DATA/data_cancer/test')

train_set = Cancer_train_dataset(train_list, train_transform, ben_prepro=ben_prepro)
val_set = Cancer_train_dataset(val_list, test_transform, ben_prepro=ben_prepro)

if Full_setting == True:
    train_list = os.listdir('/DATA/data_cancer/train')
    train_set = Cancer_train_dataset(train_list, train_transform, ben_prepro=ben_prepro)

test_set = Cancer_test_dataset(test_list, test_transform, ben_prepro=ben_prepro)

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)



print(f'num_train : {len(train_loader)}, num_val : {len(val_loader)}, num_test : {len(test_loader)}')

model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)

model.load_state_dict(torch.load('./weights/pretrain_model.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

weight_decay = 1e-5
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

print(f'n_epoch:{n_epochs}, lr:{lr}, batch_size:{batch_size}')

trial = 'best_effi-pretrain2'

best_model, best_epoch = train(model, n_cls, n_epochs, train_loader, val_loader, criterion, optimizer, scheduler, trial, device)

#torch.save(best_model.state_dict(), './weights/final_model.pth')


print("done")

