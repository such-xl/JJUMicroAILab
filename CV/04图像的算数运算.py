# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:48:45 2022

@author: xlfms
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

rain = cv.imread("./resource/img/rain.jpg")
plt.imshow(rain[:, :, ::-1])

view = cv.imread("./resource/img/view.jpg")
plt.imshow(view[:, :, ::-1])

# cv加法
img1 = cv.add(rain, view)
plt.imshow(img1[:, :, ::-1])

# cv减法
img1_1 = cv.subtract(img1, rain)
img1_2 = cv.subtract(img1, view)
plt.imshow(img1_1[:, :, ::-1])
plt.imshow(img1_2[:, :, ::-1])


# numpy 加法
img2 = rain+view
plt.imshow(img2[:, :, ::-1])

# 图像的混合
img3 = cv.addWeighted(view, 0.7, rain, 0.3, 0)
plt.imshow(img3[:, :, ::-1])
