import os
import time

import csv
import cv2
import numpy as np
import pickle
from pprint import pprint
from scipy.cluster.vq import *
from sklearn.cluster import KMeans, MiniBatchKMeans

from descriptor import Descriptor
from image_descriptor import ImageDescriptors

# file paths
PATH = './'
IMG_DIR = './imgs'
TEST_IMG_PATH = os.path.join(IMG_DIR, 'test')
TRAIN_IMG_PATH = os.path.join(IMG_DIR, 'train')
TRAIN_FEATURES_CSV = os.path.join(PATH, 'train', 'features.csv')
TEST_FEATURES_CSV = os.path.join(PATH, 'test', 'features.csv')


# global variables
labeled_imgs = {}
k_means = None

# parameters for kMeans
k = 32
batch_size = 20000
iterations = 30

des_dim = 128

# Function to write k_means object to file
def saveToFile(obj):
    with open(os.path.join(PATH, 'k_means.obj'), 'wb') as fp:
        pickle.dump(obj, fp)

def getLabels(train_img_path):
	"""Create dictionary of labeled_imgs

	Args:
	    train_img_path: path to training data
	Returns:
	    dictionary with img filenames as keys and labels as values
	Raises:
		IOError: no such directory

	"""
	try:
		for label in os.listdir(train_img_path):
			label_path = os.path.join(train_img_path, label)

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

def getVlad(imageDescriptors):
	# Set width and height
    width = imageDescriptors.width
    height = imageDescriptors.height

    # calculate width and height step
    widthStep = int(width / 2)
    heightStep = int(height / 2)

    descriptors = imageDescriptors.descriptors

    # level 1, a list with size = 4 to store histograms at different location
    VLADOfLevelOne = np.zeros((4, k, des_dim))
    for descriptor in descriptors:
        x = descriptor.x_coord
        y = descriptor.y_coord
        boundaryIndex = int(x / widthStep)  + int(y / heightStep)

        feature = descriptor.descriptor
        shape = feature.shape[0]
        feature = feature.reshape(1, shape)

        codes, distance = vq(feature, k_means.cluster_centers_)
        
        VLADOfLevelOne[boundaryIndex][codes[0]] += np.array(feature).reshape(shape) - k_means.cluster_centers_[codes[0]]
    
    
    for i in xrange(4):
        # Square root norm
        VLADOfLevelOne[i] = np.sign(VLADOfLevelOne[i]) * np.sqrt(np.abs(VLADOfLevelOne[i]))
        # Local L2 norm
        vector_norm = np.linalg.norm(VLADOfLevelOne[i], axis = 1)
        vector_norm[vector_norm < 1] = 1
        
        VLADOfLevelOne[i] /= vector_norm[:, None]
    
    # level 0
    VLADOfLevelZero = VLADOfLevelOne[0] + VLADOfLevelOne[1] + VLADOfLevelOne[2] + VLADOfLevelOne[3]
    # Square root norm
    VLADOfLevelZero = np.sign(VLADOfLevelZero) * np.sqrt(np.abs(VLADOfLevelZero))
    # Local L2 norm
    vector_norm = np.linalg.norm(VLADOfLevelZero, axis = 1)
    vector_norm[vector_norm < 1] = 1
    
    VLADOfLevelZero /= vector_norm[:, None]

    tempZero = VLADOfLevelZero.flatten() * 0.5
    tempOne = VLADOfLevelOne.flatten() * 0.5
    result = np.concatenate((tempZero, tempOne))
    # Global L2 norm
    norm = np.linalg.norm(result)
    if norm > 1.0:
        result /= norm
    return result


def getTrainingFeatures(path):
	getLabels(path)
	features = []
	keypoints = []
	filenames = labeled_imgs.keys()[:5]
	images = []
	global k_means
	for fname in filenames:
		label = labeled_imgs[fname]
		file_path = os.path.join(path, label, fname)
		kp, des = getSIFT(file_path)
		descriptors = []
		for i in xrange(len(kp)):
			x, y = kp[i].pt
			descriptor = Descriptor(x, y, des[i])
			descriptors.append(descriptor)
		imageDescriptors = ImageDescriptors(descriptors, fname, 640, 480)
		keypoints += des.tolist()
		images.append(imageDescriptors)
	print "%d keypoints extracted from the training set" % len(keypoints)
	start = time.clock()
	k_means = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, n_init=iterations)
	k_means.fit(keypoints)
	end = time.clock()
	print "Time for running %d iterations of K means for %d samples = %f" % (iterations, len(keypoints), end - start)
	
	# picklng k means
	saveToFile(k_means)
	for image in images:
		vlad = getVlad(image)
		features.append((image.filename, vlad))

	with open(TRAIN_FEATURES_CSV, 'wb') as f:
		row_writer = csv.writer(f, delimiter=',')
		for filename, vlad in features:
			row_info  = [filename]+[labeled_imgs[filename]]+vlad.tolist()
			row_writer.writerow(row_info)


if __name__ == '__main__':

	getTrainingFeatures(TRAIN_IMG_PATH)