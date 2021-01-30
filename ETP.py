# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:43:36 2020

@author: Jack
"""
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt


input_file = 'D:/UofSC_data/frames/data2new/b16/'
mask_file = 'D:/UofSC_data/cclabeler-master/data/mask2_train.jpg'

def get_image_pathes(root_path):
    img_pathes = []
    for img_path in glob.glob(os.path.join(root_path, '*.jpg')):
        img_pathes.append(img_path)
    return img_pathes

'''
# mask_train for data1new
# contours = np.array([[148, 122], [238, 80], [238, 38], [148, 38]])
# mask_train for data2new
# contours = np.array([[170, 230], [260, 200], [260, 175], [170, 175]])
# mask for data3new
contours = np.array([[168, 141], [250, 100], [250, 74], [168, 80]])

img = cv2.imread(input_file)
mask = np.full(img.shape, 0, dtype = np.uint8)
cv2.drawContours(mask, [contours], -1, (255, 255, 255), 1)

for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
        if cv2.pointPolygonTest(contours, (y, x), False) >= 0:
            mask[x, y] = (255, 255, 255)

cv2.imwrite('D:/UofSC_data/cclabeler-master/data/mask3_train.jpg', mask)
'''

k = 9
alpha = 0.1
running_avg = None
runaverage_fractions = []
motion_thresh = 20

mask = cv2.imread(mask_file)
img_pathes = get_image_pathes(input_file)
index = []
subtract = []
original_fractions = []
gray_imgs = []

for path in img_pathes:
    for i in range(len(path)-1, -1, -1):
        if path[i] == '\\': 
            index.append(int(path[i+5:-4]))
            break
    
ind_all = sorted(zip(index, img_pathes))
index = [x for x, y in ind_all]      
img_pathes = [y for x, y in ind_all]     

i = 0


for path in img_pathes:
    i += 1
    img = cv2.imread(path)
    res = cv2.bitwise_and(img, mask)

    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # mask1 ROI
    # gray_res = gray_res[38:122, 148:238]
    # mask2 ROI
    gray_res = gray_res[175:230, 170:260]
    # maks3 ROI
    #gray_res = gray_res[74:141 ,168:250] 

    #subtract.append(gray_res)
    smooth_frame = cv2.GaussianBlur(gray_res, (k, k), 0)
    
    # If this is the first frame, making running avg current frame
    # Otherwise, update running average with new smooth frame
    
    if i == 1:
        backg_avg = np.float32(smooth_frame)
    else:    
        if runaverage_fractions[-1] < 0.001:
            backg_avg = (backg_avg + np.float32(smooth_frame))/2
    
    
    if running_avg is None:
        running_avg = np.float32(smooth_frame) 
    
    # Find |difference| between current smoothed frame and running average
    diff1 = cv2.absdiff(np.float32(smooth_frame), np.float32(running_avg))
    diff2 = cv2.absdiff(np.float32(smooth_frame), np.float32(backg_avg))
    diff = (0.7*diff1) + (0.3*diff2)
    
    cv2.accumulateWeighted(np.float32(smooth_frame), running_avg, alpha)
    
    _, subtracted = cv2.threshold(diff, motion_thresh, 1, cv2.THRESH_BINARY)
        
    
    motion_fraction = (sum(sum(subtracted))/
                       (subtracted.shape[0]*subtracted.shape[1]))
    original_fractions.append(motion_fraction)
    if len(runaverage_fractions) < 5:
        motion_frion = (sum(runaverage_fractions)+motion_fraction)/(len(runaverage_fractions)+1)
    else:
        motion_fraction = (sum(runaverage_fractions[-5:])+motion_fraction)/6
    runaverage_fractions.append(motion_fraction)

print(len(runaverage_fractions))
print(len([i for i in runaverage_fractions if i > 0.025]))
print(len([i for i in original_fractions if i > 0.025]))


#plt.rc('font', size=12)          # controls default text sizes
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

plt.text(220, 0.275, 'GT: 617s', fontsize = 12)
plt.text(220, 0.255, 'ETP: 620s', fontsize = 12)
plt.plot(original_fractions, label = 'Original fraction')
plt.plot(runaverage_fractions, label = 'Smoothed fraction')
plt.axhline(y=0.025, color = 'g', label = 'Threshold', linestyle='dashed')
plt.xlabel('Frame Number')
plt.ylabel('Fraction of Motion Pixels')
#plt.title('Fraction of frame in motion over time')
plt.legend()
#plt.grid()
plt.savefig("E:/UofSC-offer/project/new_figures/passing_2.jpeg",bbox_inches = 'tight', dpi = 300)
plt.show()


