import cv2 as cv
import matplotlib.pyplot as plt
#1读取图像
img = cv.imread('resource/img/captain.jpg',0)
#2显示图像
#2.1 opencv显示
cv.imshow("captain",img)
cv.waitKey(0)
cv.destroyAllWindows()
#2.2matplotlib
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

#3 图像保存
cv.imwrite("resource/img/captain.png",img)