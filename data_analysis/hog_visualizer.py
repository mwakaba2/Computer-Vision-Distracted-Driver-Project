import sys
import os

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, io

IMGS_DIR = 'imgs/train/'
subject = 'p012'
images = [
			'img_87852.jpg', 'img_17460.jpg', 'img_36123.jpg', 'img_40181.jpg',
			'img_11090.jpg', 'img_83420.jpg', 'img_47609.jpg', 'img_82992.jpg', 
			'img_52606.jpg', 'img_33942.jpg'
			]

classes = [ 'safe driving', 
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


fig, ax = plt.subplots(10, 2, figsize=(8, 24), sharex=True, sharey=True)

for i, image in enumerate(images):
	label = 'c' + `i`
	img_path = os.path.join(IMGS_DIR, label, image)
	print "Analyzing %s" % img_path
	image = color.rgb2gray(io.imread(img_path))

	# fd: 1d flattened array
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
	print("Feature vector length: %d" % len(fd))

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

	ax[i][0].axis('off')
	ax[i][0].set_title(classes[i])
	ax[i][0].imshow(image, aspect='auto', cmap=plt.cm.gray)

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

	ax[i][1].axis('off')
	ax[i][1].set_title('%s hog' % classes[i])
	ax[i][1].imshow(hog_image_rescaled, aspect='auto', cmap=plt.cm.gray)

fig.subplots_adjust(wspace=0.1, hspace=0.3)
plt.savefig('data_analysis/output/hog_images.png', bbox_inches='tight')