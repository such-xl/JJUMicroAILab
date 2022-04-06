# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:48:45 2022

@author: xlfms
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

rain = cv.imread("./resource/img/rain.jpg")
plt.imshow(rain[:,:,::-1])
plt.show()

view = cv.imread("./resource/img/view.jpg")
plt.imshow(view[:,:,::-1])
plt.show()

#cv加法
img1 = cv.add(rain,view)
plt.imshow(img1[:,:,::-1])
plt.show()

#cv减法
img1_1 = cv.subtract(img1, rain)
img1_2 = cv.subtract(img1, view)
plt.imshow(img1_1[:,:,::-1])
plt.show()
plt.imshow(img1_2[:,:,::-1])
plt.show()

#numpy 加法
img2 = rain+view
plt.imshow(img2[:,:,::-1])
plt.show()



