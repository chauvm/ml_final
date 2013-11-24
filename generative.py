"""
Final Project
"""

import numpy as np
import matplotlib.pyplot as pyplot
sample_size = 20000

#############  Read reduced images and labels  ################
f_img_A = "trainingImagesA_reduced.npy"
f_img_B = "trainingImagesB_reduced.npy"
f_label_A = np.array(np.load("trainingLabelsA.npy", mmap_mode='r'))
f_label_B = np.array(np.load("trainingLabelsB.npy", mmap_mode='r'))
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
        print "Sigma: ", Sigma
        denom = np.sqrt(2*np.pi)**img_size/np.sqrt(np.linalg.det(Sigma))
        return np.exp(-0.5*(x-mu).T.dot(np.linalg.inv(Sigma)).dot(x-mu))/denom


#################################
################   11/20
#################################

############# prob_mixture #######################
def prob_mixture(gausses, p_vector, image):
    """
     Likelihood of image drawn from 10 Gaussians with p_vector
     (think of a better name later)
    """
    # log sum of p*Prob(image comes from each Gaussian)
    return np.log(sum([p_vector[i]*multinormalpdf(gausses[i], image) for i in range(2,10)]))

############# Gradient ###########################
def gradient(function, gausses, p_vector, image, delta):
    print "Num. Gausses : ", len(gausses)
    print "Num. p vector : ", len(p_vector)
    dims = len(p_vector)
    grad = np.zeros(dims)  
    tmp = np.zeros(dims)    
    for i in xrange(dims):     
        tmp[i] = delta
        grad[i] = (function(gausses, (p_vector+ tmp), image) - function(gausses, (p_vector- tmp), image))/delta
        tmp[i] = 0
    return grad
    
############# Normalize p_vector #################
def normalize_pvector(p_vector, gradient, delta):
    """
        Given p_vector and its gradient, return new p_vector with sum(p_vector) = 1
    """
    new_pvector = p_vector + gradient*delta
    return new_pvector/sum(new_pvector)
############# Gradient Descent ###################
def grad_desc(model, init, image):
        """
                given (gaussian) model, init prob distribution, return new prob distribution
        """
        params = init
        delta = 0.005
        acceptance = 0.01
        while (True):
                ## loss = sum of log likelihood of G(i, image)    
                # why you called it 'loss'? sum of log or log of sum
                #loss = sum([log(multinormalpdf(model[i],image)) for i in range(10)])
                # gradient should return gradients for 10 p's
                grad = gradient(prob_mixture, model, params, image, delta)
                new_params = normalize_pvector(params, grad, delta)
                
                # make use of dist function, if params do not change significantly, stop iteration
                if dist(new_params, params) < acceptance:
                    break
                else:
                    params = new_params
        return new_params

#################################
################  End of 11/20
#################################

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
print len(img_A)
print len(f_label_A)
for i in range(10):
	"""
		fit gaussian model for each 0<=i<=9, store in gauss
	"""
	batch = [img_A[j] for j in range(sample_size) if f_label_A[j]==i]

	## gauss has 10 tuples of (mean, variance)
	gauss.append(gaussian_fit(batch))
print "Num gauss at beginning: ", len(gauss)
for i in range(sample_size):
	if generative(gauss,img_B[i])!= f_label_B[i]:
		error += 1
print error/(sample_size+0.0)
