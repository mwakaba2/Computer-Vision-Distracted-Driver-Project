import sys
import os

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, io

if len(sys.argv) < 2:
	print 'usage: %s img1' % sys.argv[0]
	sys.exit(1)

img_path = sys.argv[1]
image = color.rgb2gray(io.imread(img_path))

# fd: 1d flattened array
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.show()