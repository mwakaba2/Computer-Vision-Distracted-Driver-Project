import os

from pprint import pprint

labeled_imgs = {}


def getLabels(train_path):
	"""Create dictionary of labeled_imgs

	Args:
	    train_path: path to training data
	Returns:
	    dictionary with img filenames as keys and labels as values
	Raises:
		IOError: no such directory

	"""
	for label in os.listdir(TRAIN_PATH):
		label_path = os.path.join(TRAIN_PATH, label)

		if os.path.isdir(label_path):

			for file_name in os.listdir(label_path):
				file_path = os.path.join(label_path, file_name)
				
				if os.path.isfile(file_path) and file_name.endswith('.jpg'):
					labeled_imgs[file_name] = label
	
	pprint(labeled_imgs)

if __name__ == '__main__':
	IMG_DIR = './imgs'
	TEST_PATH = os.path.join(IMG_DIR, 'test')
	TRAIN_PATH = os.path.join(IMG_DIR, 'train')

	getLabels(TRAIN_PATH)