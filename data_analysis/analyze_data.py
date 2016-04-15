import os


def analyze():
	'''
	Stats about the dataset
	'''
	IMG_DIR = './imgs'
	TEST_PATH = os.path.join(IMG_DIR, 'test')
	TRAIN_PATH = os.path.join(IMG_DIR, 'train')
	
	test_num = len([name for name in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, name))])

	print "Test data: %d files" % test_num

	for category in os.listdir(TRAIN_PATH):
		category_path = os.path.join(TRAIN_PATH, category)

		if os.path.isdir(category_path):
			category_num = len([name for name in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, name))])
			print "Category %s: %d files" % (category, category_num)

if __name__ == '__main__':
	analyze()