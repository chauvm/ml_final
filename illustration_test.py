import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

t = time.time()
sample_size = 20000

#############   Read reduced images and labels  ################
f_test = "testingImages.npy"
img_test = np.array(np.load(f_test, mmap_mode='r'))
for i in range(20):
	"""
		to get distinct images for 10 digits
	"""
	plt.imshow(img_test[i],cmap = cm.Greys_r)
	plt.show()