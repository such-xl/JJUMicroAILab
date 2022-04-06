# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:53:11 2022

@author: xlfms
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

kids = cv.imread("./resource/img/kids.jpg")
plt.imshow(kids[:,:,::-1])

#绝对尺寸
rows, cols = kids.shape[:2]
res = cv.resize(kids,(2*cols,2*rows))
plt.imshow(res[:,:,::-1])
res.shape

#相对坐标

res1 = cv.resize(kids,None,fx=0.5,fy=0.5)
plt.imshow(res1[:,:,::-1])

res1.shape

#图像平移
M = np.float32([[1,0,100],[0,1,50]])
res2 = cv.warpAffine(kids,M,(2*cols,2*rows))
plt.imshow(res2[:,:,::-1])

#图像旋转
M = cv.getRotationMatrix2D((cols/2,rows/2),45,0.5)
res3 = cv.warpAffine(kids,M,(cols,rows))
plt.imshow(res3[:,:,::-1])

#仿射变换
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1, pts2)

res4 = cv.warpAffine(kids, M,(cols,rows))
plt.imshow(res4[:,:,::-1])

#透视变换
pst1 = np.float32([[56,65],[368,52],[28,387],[398,390]])
pst2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
T = cv.getPerspectiveTransform(pst1, pst2)
res4 = cv.warpPerspective(kids,T,(cols,rows))
plt.imshow(res4[:,:,::-1])

#图像金字塔
imgup = cv.pyrUp(kids)
plt.imshow(imgup[:,:,::-1])
imgup2 = cv.pyrUp(imgup)
plt.imshow(imgup2[:,:,::-1])

imgdown = cv.pyrDown(kids)
plt.imshow(imgdown[:,:,::-1])
imgdown2 = cv.pyrDown(imgdown)
plt.imshow(imgdown2[:,:,::-1])