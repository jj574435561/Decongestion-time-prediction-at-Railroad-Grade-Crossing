#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:23:28 2020

@author: zhuochengjiang
"""

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from torchvision import datasets, transforms
import cv2


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
                   ])

root = '/home/zhuochengjiang/CSRNet-pytorch/b27_ROI/'

def get_json_pathes(root_path):
    img_pathes = []
    for img_path in glob.glob(os.path.join(root_path, '*.jpg')):
        img_pathes.append(img_path)
    return img_pathes

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('/home/zhuochengjiang/0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
img_paths = get_json_pathes(root)
index = []
result = []
for path in img_paths:
    
    img = transform(Image.open(path).convert('RGB')).cuda()
    for i in range(len(path)-1, -1,-1):
        if path[i] == '/':
            index.append(int(path[i+5:-4]))
            break
    #print(index)
    
    output = model(img.unsqueeze(0))
    output = output.detach().cpu().numpy().squeeze()
    count = np.sum(output)
    result.append(count)
    #print(index[-1], count)



ind_cou = sorted(zip(index, result))
index_sort = [x for x,y in ind_cou]
result_sort = [y for x, y in ind_cou]
raise_up = 538

#plt.plot(index_sort, result_sort, markevery = mark, color = 'red', marker = 'd', markersize = 12)
plt.plot(index_sort, result_sort)
print(np.average(result_sort[:100]), np.max(result_sort))
plt.scatter(index_sort[100+raise_up], result_sort[100+raise_up], color = 'red')
plt.xlabel('Frames')
plt.ylabel('Number of Cars')
plt.show()
    
    
    
    
    
    
    