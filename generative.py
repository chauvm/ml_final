"""
Final Project
"""

import numpy as np
import matploylib.pyplot as pyplot
sample_size = 20000

#############Read reduced images and labels################
f_img_A = "trainingImagesA_red.npy"
f_img_B = "trainingImagesB_red.npy"
f_label_A = "trainingLabelsA.npy"
f_label_B = "trainingLabelsB.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
img_B = np.array(np.load(f_img_B, mmap_mode='r'))

#############Gradient Descent###################
def grad_desc():

#############Fit Gaussian model#################
def gaussian_fit(images):
	mean = sum([image for image in images])/(len(images)+i)
	variance = grad_desc(images, init)
	return (mean, variance)

#############Classifier#########################
def generative(gauss,image):


#############Execution##########################
gauss = []
for i in range(10):
	"""
		fit gaussian model for each 0<=i<=9, store in gauss
	"""
	batch = [images_A[j] for j in range(sample_size) if label_A[j]==i]
	np.append(gauss,[gaussian_fit(batch)])

for i in range(sample_size):
	if generative(gauss,images_B[i])!=label_B[i]:
		error += 1
print error/(sample_size+0.0)