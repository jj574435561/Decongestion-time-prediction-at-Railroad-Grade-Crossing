# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:58:00 2020

@author: Jack

"""

import sys
import os
from model import CSRNet
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
import dataset
import time
from matplotlib import pyplot as plt

root = '/home/zhuochengjiang/CSRNet-pytorch/test_paths.json'

model = CSRNet()
model = model.cuda()

with open(root, 'r') as outfile:        
    val_list = json.load(outfile)
test_loader = torch.utils.data.DataLoader(
dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
                   ]),  train=False),
    batch_size=1, shuffle = False)    
    
mae = 0
avg = 0
    
for i, (img, target) in enumerate(test_loader):
    img = img.cuda()
    img = Variable(img)
    output = model(img)
    
    mae = output.data.sum()-target.sum().type(torch.FloatTensor).cuda()
    avg += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
    print('MAE {mae:.3f}'.format(mae = mae))
    if i == 2:
        plt.imshow(output)
        plt.show()
avg = avg/len(test_loader)    
print(' * AVG_MAE {mae:.3f} '
              .format(mae=avg))


