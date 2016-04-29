import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imgs/train/c3/img_10034.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
cv2.imshow('image',img)
plt.show()