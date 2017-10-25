# _*_ coding:utf-8 _*_
import cv2
import numpy as np

image=cv2.imread("data/img.jpg")
cv2.imshow('original',image)

kernel=np.array([
    [.11,.11,.11],
    [.11,.11,.11],
    [.11,.11,.11]
                 ])
rect=cv2.filter2D(image,-1,kernel)
cv2.imwrite('rect.jpg',rect)

