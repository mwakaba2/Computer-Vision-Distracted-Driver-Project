import os

import cv2
import numpy as np
from pprint import pprint


# global variables
labeled_imgs = {}
k_means = None

def getLabels(train_path):
	"""Create dictionary of labeled_imgs

	Args:
	    train_path: path to training data
	Returns:
	    dictionary with img filenames as keys and labels as values
	Raises:
		IOError: no such directory

	"""
	try:
		for label in os.listdir(train_path):
			label_path = os.path.join(train_path, label)

			if os.path.isdir(label_path):

				for file_name in os.listdir(label_path):
					file_path = os.path.join(label_path, file_name)
					
					if os.path.isfile(file_path) and file_name.endswith('.jpg'):
						labeled_imgs[file_name] = label
	except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
	
	# pprint(labeled_imgs)

def getSIFT(path):
	img = cv2.imread(path)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.SIFT()
	key_pts, des = sift.detectAndCompute(gray_img, None)

	return key_pts, des

def getTrainingFeatures(path):
	getLabels(path)
	features = []
	sd = []
	filename = labeled_imgs.keys()[0]
	images = []
	global k_means
	file_path = os.path.join(path, labeled_imgs[filename], filename)

	kp, des = getSIFT(file_path)
	
	print kp[0].pt, des[0]


if __name__ == '__main__':
	IMG_DIR = './imgs'
	TEST_PATH = os.path.join(IMG_DIR, 'test')
	TRAIN_PATH = os.path.join(IMG_DIR, 'train')

	getTrainingFeatures(TRAIN_PATH)
	#getLabels(TRAIN_PATH)