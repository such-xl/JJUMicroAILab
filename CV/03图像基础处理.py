import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = np.zeros((256,256,3),np.uint8)

plt.imshow(img[:,:,::-1])
print(img[100,100])
print(img[100,100,0])
img[100,100] = (0,0,255)
plt.imshow(img[:,:,::-1])
print(img.shape)
print(img.dtype)
print(img.size)

#图像通道的拆分与合并
captain = cv.imread("CV/resource/img/captain.jpg")

plt.imshow(captain[:,:,::-1])
b,g,r = cv.split(captain)
plt.imshow(b,cmap=plt.cm.gray)
img2 = cv.merge((b,g,r))
plt.imshow(img2[:,:,::-1])
#色彩空间的改变
gray = cv.cvtColor(captain,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap=plt.cm.gray)
hsv = cv.cvtColor(captain,cv.COLOR_BGR2HSV)
plt.imshow(hsv)
