"""
Final Project
"""

import numpy as np
import matplotlib.pyplot as pyplot
import time

t = time.time()
sample_size = 20000

#############   Read reduced images and labels  ################
f_img_A = "trainingImagesA_reduced.npy"
f_img_B = "trainingImagesB_reduced.npy"
f_label_A = "trainingLabelsA.npy"
label_A = np.array(np.load(f_label_A, mmap_mode='r'))
f_label_B = "trainingLabelsB.npy"
label_B = np.array(np.load(f_label_B, mmap_mode='r'))
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
img_B = np.array(np.load(f_img_B, mmap_mode='r'))
img_size = len(img_A[0])**2

img_train = img_A[0:3000]
label_train = label_A[0:3000]
# img_train = img_B[0:1500]
# label_train = label_B[0:1500]

############# Fit Gaussian model #################
def gaussian_fit(images):
    print "Shape of images: ", images.shape
    mean = sum([image for image in images])/(len(images)+0.0)
    print "Mean of images: ", mean
    variance = images.T.dot(images)/(len(images)-1.0)
    print "Variance: ", variance.shape
    zero_rows = []
    for i in range(len(variance)):
        if np.all(variance[i]==0):
            zero_rows.append(i)
    new_mean = np.delete(mean, zero_rows, 0)
    #new_mean = np.delete(_mean, zero_rows, 1)
    
    _variance = np.delete(variance, zero_rows, 0)
    new_variance = np.delete(_variance, zero_rows, 1)
    
    ind_mean_dict = {}
    for ind in zero_rows:
        ind_mean_dict[ind] = mean[ind]
        print "New variance: ", new_variance
    return (new_mean, new_variance, ind_mean_dict)

############# Multivariate normal density ########
def multinormalpdf((mu, Sigma, ind_mean_dict), x):
        #print "Sigma: ", Sigma
        
        for ind in ind_mean_dict.keys():
                if np.all((x[ind] - ind_mean_dict[ind]) == 0):
                        return 0
        new_x = np.delete(x, ind_mean_dict.keys(), 0)
        #new_x = np.delete(_x, ind_mean_dict.keys(), 1)
        #print "Determinant of Sigma: ", np.sqrt(np.linalg.det(Sigma))
        #denom = ((np.sqrt(2*np.pi))**img_size)*np.sqrt(np.linalg.det(Sigma))
        denom = np.sqrt(np.linalg.det(Sigma))
        #print "Denominator in multinormalpdf: ", denom
        prob = np.exp(-0.5*(new_x-mu).T.dot(np.linalg.inv(Sigma)).dot(new_x-mu))/denom
#       inex = np.dot(np.linalg.inv(Sigma),(new_x-mu).T)
#       print "Inex: ", inex
#       sec_inex = np.dot((new_x-mu),inex)
#       print "Second inex: ", sec_inex
#       prob = np.exp(-0.5*sec_inex)/denom
        
        #print "Prob of a multinormalpdf:", prob
        return prob


############# Training error  #######################
def loss(mixtures):
    """
     Training error based on current mixture proportions
     Loss = sum over training set: [[label != prediction]]
    """
    ## image_A or img_train? img_train for now
    loss = sum([1 for i in range(len(img_train)) if generative(gausses, mixtures, img_train[i])!=label_train[i]])
    return loss/(len(img_train)+0.0)

############# Training error using Square loss  #######################
def squareloss(mixtures):
    """
     Training error based on current mixture proportions
     Loss = sum over training set: [[label != prediction]]
    """
    ## image_A or img_train? img_train for now
    loss = sum([(generative(gausses, mixtures, img_train[i])- label_train[i])**2 for i in range(len(img_train))])
    return loss/(len(img_train)+0.0)

############# Gradient ###########################
def gradient(function, mixtures, delta):
    dims = len(mixtures)
    grad = np.zeros(dims)   
    tmp = np.zeros(dims)        
    for i in xrange(dims):       
        tmp[i] = delta
        grad[i] = (function(mixtures+ tmp) - function(mixtures- tmp))/delta
        tmp[i] = 0
    print "Gradient: ", grad
    return grad
    
############# Normalize p_vector #################
def normalize_mixtures(p_vector, current_loss, gradient, eta):
    """
        Given p_vector and its gradient, return new p_vector with sum(p_vector) = 1
    """
    new_pvector = p_vector - gradient*eta
    # if squareloss(new_pvector) > current_loss:
    #         new_pvector = p_vector + gradient*eta
    #         if squareloss(new_pvector) > current_loss:
    #                 print "Bad gradient"
    return new_pvector/sum(new_pvector)

############### OLD Gradient Descent ###################
##def grad_desc(models, init, images, labels):
##        """
##            given (gaussian) model, init prob distribution, return new prob distribution
##            compare (loss1+1)/(loss2+1), if really close to 1, stop
##                
##        """
##        params = init
##        delta = 0.05
##        acceptance = 10**6
##        for i in range(2):
##            grad = gradient(loss, models, params, images, delta)
##            new_params = normalize_pvector(params, grad, delta)
##            
##            # make use of dist function, if params do not change significantly, stop iteration
##            if dist(new_params, params) < acceptance:
##                break
##            else:
##                print "Distance: ", dist(new_params, params)
##                params = new_params
##        return params

############# Gradient Descent ###################
squareLosses = []
Losses = []
def grad_desc(mixtures, delta = 0.006):
        """
            given (gaussian) model, init prob distribution, return new prob distribution
            compare (loss1+1)/(loss2+1), if really close to 1, stop
                
        """
        squareLosses.append(squareloss(mixtures))
        Losses.append(loss(mixtures))
        
        for i in range(15):
            grad = gradient(loss, mixtures, delta)
            mixtures = normalize_mixtures(mixtures, Losses[-1], grad, 0.0015) #0.000001
            #squareLosses.append(squareloss(mixtures))            
            Losses.append(loss(mixtures))
            print "Mixtures: ", mixtures
            #print "squareLosses: ", squareLosses            
            print "Losses: ", Losses
        # pyplot.plot(Losses)
        # pyplot.show()
        return mixtures
############# Classifier #########################
def generative(gausses,mixtures, image):
        """
                given the image, return label with max prob
        """
        #p_vector = np.array([mixtures[i]*multinormalpdf(gausses[i],image) for i in range(10)])
        p_vector = np.array([mixtures[i]*np.log(multinormalpdf(gausses[i],image)) for i in range(10)])
        
        #so p_vector = [multinormalpdf(gauss[i],image) for i in range(10)]
        return np.argmax(p_vector)
        
#################Train mixtures###################
init = np.array([0.1]*10)
#mixtures = grad_desc(gausses, init, image_train, label_train)

############# Execution: Train data using img_train ##########################

gausses = []
print "Number of images in training data: ", len(img_A)
batches = []

### need to refractor this. basically you're iterating over 2000 images 10 times!
for i in range(10):
    """
        fit gaussian model for each 0<=i<=9, store in gauss
    """
    batch = np.array([img_A[j] for j in range(len(img_A)) if label_A[j]==i])
    batches.append(batch.shape[0])

    ## gauss has 10 lists of (mean, variance, {indices of 0s: entries of mean})
    gausses.append(gaussian_fit(batch))
error = 0

batches = np.array(batches)
#mixtures = batches/(sum(batches)+0.0)
mixtures = [0.1]*10
print "Initial mixtures: ", mixtures
print loss(mixtures)

#### After this, we have gausses, initial mixtures and its initial loss
#### Now use gradient descent to minimize loss i.e varying mixtures
mixtures = grad_desc(mixtures)

for i in range(1,sample_size):
	if generative(gausses, mixtures, img_B[i])!= label_B[i]:
		error += 1
	if i%500 == 0:
		print "After ", i, " test points, error = ", error/(i+0.0)
		print time.time()-t
print error/(sample_size+0.0)
