# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:14:32 2022

@author: xlfms
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

girlSp = cv.imread("./resource/img/girlSp.png")
girlGauss = cv.imread("./resource/img/girlGauss.jpg")
plt.imshow(girlSp[:,:,::-1])
plt.imshow(girlGauss[:,:,::-1])
#均值滤波
girl1 = cv.blur(girlSp,(11,11))
plt.imshow(girl1[:,:,::-1])

#高斯滤波
girl2 = cv.GaussianBlur(girlGauss, (3,3),1)
plt.imshow(girl2[:,:,::-1])

#中值滤波
girl3 = cv.medianBlur(girlSp,15)
plt.imshow(girl3[:,:,::-1])