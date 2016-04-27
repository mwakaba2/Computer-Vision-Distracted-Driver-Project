import os
import time

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pylab as pl
from pprint import pprint
from scipy.cluster.vq import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.pipeline import Pipeline
from descriptor import Descriptor
from image_descriptor import ImageDescriptors

# file paths
PATH = './'
IMG_DIR = './imgs'
TEST_IMG_PATH = os.path.join(IMG_DIR, 'test')
TRAIN_IMG_PATH = os.path.join(IMG_DIR, 'train')
TRAIN_FEATURES_CSV = os.path.join(PATH, 'train', 'features.csv')
TEST_OUTPUT = os.path.join('./test', "test.csv")

# global variables
labeled_imgs = {}
k_means = None

# parameters for kMeans
k = 32
batch_size = 20000
iterations = 30

des_dim = 32

# Function to write k_means object to file
def saveToFile(obj):
    with open(os.path.join(PATH, 'k_means.obj'), 'wb') as fp:
        pickle.dump(obj, fp)

def loadFromFile():
    with open(os.path.join(PATH, 'k_means.obj'), 'rb') as fp:
        return pickle.load(fp)

def readCSV(filename, is_train):
	fileNames = []
	X = []
	labels = []
	with open(filename, 'rb') as f:
		reader = csv.reader(f, delimiter = ',')
        
		for row in reader:
			fileNames.append(row[0])
			if is_train:
				features = [float(ele) for ele in row[2:]]
			else:
				features = [float(ele) for ele in row[1:]]
			X.append(features)
            
			if is_train:
				output_class = row[1].strip(' \n\t\r')
				labels.append(output_class)

	return fileNames, labels, X

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

def getSURF(fileName):
    # read image
    img = cv2.imread(fileName)
    
    # convert image to gray scale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    '''
    creates surf detector
    hessianThreshold - determines no. of detected keypoints (lower the threshold, more the number of keypoints)
    value set to 500 based on suggestions in paper
    extended - if set to True, computes SURF of 128 dimension; otherwise 64 dimension
    '''
    detector = cv2.SURF(hessianThreshold = 500, extended = True)
    
    # get keypoints and descriptors for them
    kp, des = detector.detectAndCompute(grey, None)
    
    # if there are no keypoints found, enhance the contrast of the image and try again
    if des.size == 0:
        equ = cv2.equalizeHist(grey)
        kp, des = detector.detectAndCompute(equ, None)
    
    return kp, des

def getSIFT(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    dense = cv2.FeatureDetector_create("Dense")
    kp = dense.detect(gray_img)
    kp, des = sift.compute(gray_img,kp)
    return kp, des

def getSTAR(path):
    img = cv2.imread(path,0)
    # Initiate STAR detector
    star = cv2.FeatureDetector_create("STAR")
    # Initiate BRIEF extractor
    brief = cv2.DescriptorExtractor_create("BRIEF")
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)
    return kp, des

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
    filenames = labeled_imgs.keys()
    images = []
    global k_means
    for fname in filenames:
        label = labeled_imgs[fname]
        file_path = os.path.join(path, label, fname)
        if os.path.isfile(file_path) and fname.endswith('.jpg'):
            kp, des = getSTAR(file_path)
            descriptors = []
            for i in xrange(len(kp)):
                x, y = kp[i].pt
                descriptor = Descriptor(x, y, des[i])
                descriptors.append(descriptor)
            imageDescriptors = ImageDescriptors(descriptors, fname, 640, 480)
            keypoints += des.tolist()
            images.append(imageDescriptors)
            print "%d images and %d keypoints" % (len(images), len(keypoints))
        else:
            continue
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

def getTestingFeatures(path):	
    features = []
    lbls = []
    images = []
    filenames = [f for f in os.listdir(TEST_IMG_PATH) if os.path.isfile(os.path.join(TEST_IMG_PATH,f))]

    for filename in filenames:
        file_path = os.path.join(path, filename)
        print "filename %s" % file_path
        if os.path.isfile(file_path) and filename.endswith('.jpg'):
            kp, des = getSTAR(file_path)

            descriptors = []
            for i in xrange(len(kp)):
                x,y = kp[i].pt
                descriptor = Descriptor(x, y, des[i])
                descriptors.append(descriptor)
	            
            imgDescriptor = ImageDescriptors(descriptors, filename, 640, 480)
            images.append(imgDescriptor)
            print "%d images" % (len(images))

    for image in images:
        vlad = getVlad(image)
        features.append(vlad.tolist())
        lbls.append(image.filename)
        print "features list: %d features" % len(features)
        # since test set is huge, need to classify images in batches of 100
        if len(features) == 61:
            print "predicting labels!"
            outputLabels(lbls, features)
            features = []
            lbls = []


def outputLabels(lbls, X):
    Y = np.array(X)
    Y = Y.astype(float)
    scale(Y, with_mean = True, with_std = True)
    preds = model.predict_proba(Y)
    writePredictions(lbls, preds)

def writePredictions(labels, preds):
    with open(TEST_OUTPUT, 'a') as fp:
         writer = csv.writer(fp, delimiter = ',')
         for label, pred in zip(labels, preds):
             writer.writerow([label]+[ '%.2f' % elem for elem in pred])

def exhaustiveGridSearch():
    n_estimators = 10
    # read training file
    lbls1, y, X = readCSV(TRAIN_FEATURES_CSV, True)

    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)

    # scale features for zero mean and unit variance 
    scale(X, with_mean = True, with_std = True)

    # Split the dataset in two parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
    # Set the parameters by cross-validation

    tuned_parameters = {
                            'estimator__n_estimators':[10, 15, 20, 25],
                            'estimator__n_jobs': [5, 10, 15, 20],
                            'estimator__max_samples':[1.0/10, 1.0/15, 1.0/20, 1.0/25],
    }
                    
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s\n" % score)
        clf = GridSearchCV(OneVsRestClassifier(BaggingClassifier(SVC(C=4.0, probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs = 15)), tuned_parameters, cv=5, scoring=score)
        #clf = GridSearchCV(RandomForestClassifier(n_estimators = 200, random_state = 100), tuned_parameters, cv=5, scoring=score)
        #clf = GridSearchCV(AdaBoostClassifier(base_estimator=svm.SVC(C=1), n_estimators = 200, random_state = 100), tuned_parameters, cv=5, scoring=score))
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:\n")
        print(clf.best_estimator_)
        print("Grid scores on development set:\n")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
        print("Detailed classification report:\n")
        print("The model is trained on the full development set.\n")
        print("The scores are computed on the full evaluation set.\n")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

# Compute Histogram Intesection kernel
def hist_intersection(x, y):
    n_samples , n_features = x.shape
    K = np.zeros(shape=(n_samples, n_samples),dtype=np.float)
    
    for r in xrange(n_samples):
        for c in xrange(n_samples):
            K[r][c] = np.sum(np.minimum(x[r], y[c]))
    return K

# Function to perform cross validation on different models
def cross_validate(X, y):
    
    svc = svm.SVC(kernel='linear', C = 0.0625)
    
    lin_svc = svm.SVC(C = 4.0, dual = False)
    
    rbf_svc = svm.SVC(kernel='rbf', gamma = 0.0009765625, C = 32.0)
    
    poly_svc = svm.SVC(kernel='poly', degree = 2 , C = 2048.0)
    
    hist_svc = svm.SVC(kernel = 'precomputed')
    chi2_svc = svm.SVC(kernel = 'precomputed')
    
    # random_forest = RandomForestClassifier(n_estimators = 200, max_features = 50, min_samples_split = 20, random_state = 100)
    # 5-fold cross validation
    for model in [svc, lin_svc, rbf_svc, poly_svc]:
        print model
        scores = cross_val_score(model, X, y, cv=10)
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print hist_svc
    K = hist_intersection(X, X)
    scores = cross_val_score(model, K, y, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print chi2_svc
    K = chi2_kernel(X, gamma = 0.3)
    scores = cross_val_score(model, K, y, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
             

# Function to perform training and classification
def classify():
    # read training data
    lbls1, X, y = readCSV(TRAIN_FEATURES_CSV, True)
    # read test data
    lbls2, Y, z = readCSV(TEST_OUTPUT, false)
    
    # Conversion to numpy arrays
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    
    Y = np.array(Y)
    Y = Y.astype(float)
    
    # perform feature scaling for zero mean and unit variance
    scale(X, with_mean = True, with_std = True)
    scale(Y, with_mean = True, with_std = True)
    
    lin_svc = svm.SVC(C = 4.0, dual = False)
    lin_svc.fit(X, y)
    
    bestmodel = lin_svc
    preds = bestmodel.predict(Y)
    
    writePredictions(lbls2, preds)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':

    #getTrainingFeatures(TRAIN_IMG_PATH)
    # print "Loading k_means..."
    # k_means = loadFromFile()
    # print "Finished loading!"
    # print "Reading CSV"
    fileName, labels, X = readCSV(TRAIN_FEATURES_CSV, True)
    print "Done reading!"
    X = np.array(X)
    labels = np.array(labels)
    scale(X, with_mean = True, with_std = True)
    n_estimators = 10
    classifier = OneVsRestClassifier(BaggingClassifier(SVC(C=4.0, probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs = 15))
    print "fitting classifier..."
    classifier.fit(X, labels)
    print "done fitting!"
    model = classifier
    #getTestingFeatures(TEST_IMG_PATH)
    
    # Split the data randomly into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=100)
    y_true, y_pred = y_test, classifier.predict(X_test)
    print(classification_report(y_true, y_pred))
    # # Run classifier
    # y_pred = classifier.predict(X_test)
    # # Compute confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # plt.figure()
    # plot_confusion_matrix(cm)

    # # Normalize the confusion matrix by row (i.e by the number of samples
    # # in each class)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Normalized confusion matrix')
    # print(cm_normalized)
    # plt.figure()
    # plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    # plt.show()

    #exhaustiveGridSearch()
    # classify()