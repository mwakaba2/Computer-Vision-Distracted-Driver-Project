

class ImageDescriptors:

	def __init__(self, descriptors, filename, width, height):
		self.descriptors = descriptors
		self.filename = filename
		self.width = width
		self.height = height

	def __str__(self):
		return "ImageDescriptors:\n %d descriptors\n filename =\n %s" % (len(self.descriptors), self.filename)
