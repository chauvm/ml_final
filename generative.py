"""
Final Project
"""

import numpy as np
import matploylib.pyplot as pyplot
sample_size = 20000

#############  Read reduced images and labels  ################
f_img_A = "trainingImagesA_red.npy"
f_img_B = "trainingImagesB_red.npy"
f_label_A = "trainingLabelsA.npy"
f_label_B = "trainingLabelsB.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
img_B = np.array(np.load(f_img_B, mmap_mode='r'))
img_size = len(img_A[0])**2

############# Fit Gaussian model #################
def gaussian_fit(images):
	mean = sum([image for image in images])/(len(images)+0.0)
	variance = image.T.dot(image)/(len(images)-1.0)
	return (mean, variance)

############# Multivariate normal density ########
def multinormalpdf(muSigma, x):
        mu, Sigma = muSigma
        return np.exp(-0.5*(x-mu).T.dot(np.linalg.inv(Sigma)).dot(x-mu))/sqrt(2*np.pi)**img_size/sqrt(np.linalg.det(Sigma))

############# Gradient Descent ###################
def grad_desc(model, init, image):
        """
                given (gaussian) model, init prob distribution, return new prob distribution
        """
        params = init
        while (True):
                loss = sum([log(multinormalpdf(model[i],image)) for i in range(10)])
                gradient = gradient(loss,,,)
                ### careful: we can't make sure probs sum to 1

############# Classifier #########################
def generative(gauss,image):
        """
                given the image, return label with max prob
        """
        p_vector = np.array([0.1 for i in range(10)])
        p_vector = grad_desc(gauss, p_vector, image)
        return p_vector.index(max(p_vector))

############# Execution ##########################
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
