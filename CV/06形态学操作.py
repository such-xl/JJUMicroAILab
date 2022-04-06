# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 20:44:39 2022

@author: xlfms
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#腐蚀与膨胀
img =cv.imread("./resource/img/letter.png")
plt.imshow(img[:,:,::-1])

#创建核结构
kenel = np.ones((5,5),np.uint8)
#腐蚀
img2  = cv.erode(img,kenel)
plt.imshow(img2[:,:,::-1])

#膨胀
img3 = cv.dilate(img,kenel)
plt.imshow(img3[:,:,::-1])

#开运算 先腐蚀,再膨胀
kenel = np.ones((20,20),np.uint8)
cvopen = cv.morphologyEx(img,cv.MORPH_OPEN,kenel)
plt.imshow(cvopen[:,:,::-1])
#闭运算 先膨胀,再腐蚀
cvclose = cv.morphologyEx(img,cv.MORPH_CLOSE,kenel)
plt.imshow(cvclose[:,:,::-1])

#礼帽(顶帽)运算, 原图像与开运算的结果之差 提取背景
top = cv.morphologyEx(img, cv.MORPH_TOPHAT, kenel)
plt.imshow(top)
#黑帽(底帽)运算 闭运算与原图之差 
black = cv.morphologyEx(img,cv.MORPH_BLACKHAT,kenel)
plt.imshow(black)

