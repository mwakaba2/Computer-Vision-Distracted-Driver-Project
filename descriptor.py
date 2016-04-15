import numpy as np

class Descriptor:

	def __init__(self, x_coord, y_coord, descriptor):
		self.x_coord = x_coord
		self.y_coord = y_coord
		self.descriptor = self.normalize(descriptor)

	def normalize(self, descriptor):
		descriptor = np.array(descriptor)
		norm = float(np.linalg.norm(descriptor))

		if norm < 1.0:
			return descriptor
		else:
			return descriptor / norm

	def __str__(self):
		return "Descriptor:\n x = %.2f\n y = %.2f\n des =\n %s" % (self.x_coord, self.y_coord, self.descriptor)
