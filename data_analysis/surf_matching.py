import os
import cv2
import sys
import scipy as sp
import numpy as np

IMGS_DIR = 'imgs/train/'
imgs1 = ['c0/img_31613.jpg', 'c1/img_70529.jpg', 'c2/img_51435.jpg', 
		'c3/img_11340.jpg', 'c4/img_13710.jpg', 'c5/img_32871.jpg',
		 'c6/img_98118.jpg', 'c7/img_53894.jpg', 'c8/img_11324.jpg', 
		 'c9/img_68251.jpg']
imgs2 = ['c0/img_98046.jpg', 'c1/img_18849.jpg', 'c2/img_85485.jpg', 
		'c3/img_73194.jpg', 'c4/img_45737.jpg', 'c5/img_43925.jpg', 
		'c6/img_55834.jpg', 'c7/img_101869.jpg', 'c8/img_75770.jpg', 
		'c9/img_20688.jpg']

labels = [ 'safe driving', 
			'texting - right', 
			'talking on the phone - right', 
			'texting - left', 
			'talking on the phone - left ',
			'operating the radio',
			'drinking',
			'reaching behind', 
			'hair and makeup',
			'talking to passenger'
			]

for img, img2, label in zip(imgs1, imgs2, labels):
	img1_path = os.path.join(IMGS_DIR, img)
	img2_path = os.path.join(IMGS_DIR, img2)

	img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

	detector = cv2.FeatureDetector_create("SURF")
	descriptor = cv2.DescriptorExtractor_create("BRIEF")
	matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

	# detect keypoints
	kp1 = detector.detect(img1)
	kp2 = detector.detect(img2)

	print '#keypoints in image1: %d, image2: %d' % (len(kp1), len(kp2))

	# descriptors
	k1, d1 = descriptor.compute(img1, kp1)
	k2, d2 = descriptor.compute(img2, kp2)

	print '#keypoints in image1: %d, image2: %d' % (len(d1), len(d2))

	# match the keypoints
	matches = matcher.match(d1, d2)

	# visualize the matches
	print '#matches:', len(matches)
	dist = [m.distance for m in matches]

	print 'distance: min: %.3f' % min(dist)
	print 'distance: mean: %.3f' % (sum(dist) / len(dist))
	print 'distance: max: %.3f' % max(dist)

	# threshold: half the mean
	thres_dist = (sum(dist) / len(dist)) * 0.50

	# keep only the reasonable matches
	sel_matches = [m for m in matches if m.distance < thres_dist]

	print '#selected matches:', len(sel_matches)

	# visualization of the matches
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
	view[:h1, :w1, :] = np.dstack([img1, img1, img1])
	view[:h2, w1:, :] = np.dstack([img2, img2, img2])

	for m in sel_matches:
	    # draw the keypoints
	    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
	    cv2.line(view, (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])) , (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)


	cv2.imshow(label, view)
	cv2.waitKey()