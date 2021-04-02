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
from Cancer_dataset import Cancer_train_dataset, Cancer_test_dataset
from torchvision import models
import torch.nn as nn
from utils import *
import torchvision
import torchvision.transforms.functional as TF
import scipy as sp
import time
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

import torchvision
import time

from classifier_utils import submit, submit_probs

seed=20
seed_everything(seed)


batch_size = 64
image_size = 456

class Test_Rotation(DualTransform):
    """for tta"""
    def __init__(self, angle, always_apply=False, p=0.5):
        self.angle = angle
        self.always_apply = always_apply
        self.p = p

        super(Test_Rotation, self).__init__(always_apply, p)
        self.mask_value = None

    def apply(self, image, **params):
        # print(image.shape)
        return sp.ndimage.rotate(image, self.angle, axes=(0,1), order=1,
                reshape=False, mode='reflect')

    def get_transform_init_args_names(self):
        return ()

class Test_Shift(DualTransform):
    """for tta"""
    def __init__(self, shift, always_apply=False, p=0.5):
        self.shift = shift + [0]
         # 0(h) or 1(w)
        self.always_apply = always_apply
        self.p = p

        super(Test_Shift, self).__init__(always_apply, p)
        self.mask_value = None

    def apply(self, image, **params):
        # print(image.shape)
        return sp.ndimage.shift(image, self.shift, order=1, mode='reflect')

    def get_transform_init_args_names(self):
        return ()


scale = A.Resize(image_size, image_size)
scale2 = A.Resize(528, 528)
#rcrop = A.RandomCrop(width=256, height=256)
shift_w = Test_Shift([0,10], p=1)
shift_h = Test_Shift([10,0], p=1)
rotate1 = Test_Rotation(15, p=1)
rotate2 = Test_Rotation(-15, p=1)
H_flip = A.HorizontalFlip(p=1)
V_flip = A.VerticalFlip(p=1)

normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
to_tensor = ToTensor()


test_list = os.listdir('/DATA/data_cancer/test')


#file_name = 'effi-b5-456-reduceLR_epoch25' # best
#file_name = 'res152-224-ssr-sgd_epoch23'
file_name = 'best_effi-pretrain_epoch10'

model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)
#model = models.resnet152(pretrained=False)
#model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load( 'model/' + file_name + '.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

make_csv = True
save_dir = './result/'
createFolder(save_dir)
print('batch_size', batch_size)
transform_list = [None, V_flip, H_flip, rotate1, rotate2, shift_w, shift_h] # 0, 1, 2, 3, 4
#transform_list = [rotate1, rotate2] 
t_name_list = ['o', 'V','H','r15','r' ,'sw10', 'sh' ]
cur_t_name = ''

def make_csv(probs, file_name, prefix='' ):
    prefix = prefix + '_tta_'
    results_df = pd.DataFrame(probs, columns = ['filename', 'pred'])
    results_df.to_csv(save_dir + prefix + file_name + '.csv', header=True, index=False)
    print('make!!')


probs = 0 # 누적 probs
print('TTA submit')
for idx, new_transform in enumerate(transform_list):
    if new_transform is None:
        new_transform = [scale]
    elif new_transform == 's2':
        new_transform = [scale2]
    else:
        new_transform = [scale, new_transform]
    cur_t_name += t_name_list[idx]
    test_transform = A.Compose( new_transform +  [normalize, to_tensor] )
    print('transform', new_transform)
    test_set = Cancer_test_dataset(test_list, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    print(f'num_test : {len(test_loader)}')

    dic = submit_probs(model, test_loader, device)
    probs += dic['probs']
    name_probs = np.stack([dic['names'], probs/(idx+1)], 1)
    print(name_probs.shape)
    make_csv(name_probs, file_name, cur_t_name)


print("done")
