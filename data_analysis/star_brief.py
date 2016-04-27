import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print 'usage: %s img1' % sys.argv[0]
	sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path, 0)	
print img
# Initiate STAR detector
star = cv2.FeatureDetector_create("STAR")
# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
print des.shape

plt.imshow(img2)
plt.show()
