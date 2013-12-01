"""
Final Project: GDA for digit recognition
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
lbl_A = np.array(np.load(f_label_A, mmap_mode='r'))
f_label_B = "trainingLabelsB.npy"
lbl_B = np.array(np.load(f_label_B, mmap_mode='r'))
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
img_B = np.array(np.load(f_img_B, mmap_mode='r'))
img_size = len(img_A[0])**2
img_train = np.append(img_A,img_B, axis=0)
lbl_train = np.append(lbl_A,lbl_B, axis=0)
# img_train = img_A
# lbl_train = lbl_A
f_test = "testingImages_red.npy"
img_test = np.array(np.load(f_test, mmap_mode='r'))

############# Fit Gaussian model #################
def gaussian_fit(images):
    # print "Shape of images: ", images.shape
    mean = sum([image for image in images])/(len(images)+0.0)
    # print "Mean of images: ", mean
    variance = images.T.dot(images)/(len(images)-1.0)
    # print "Variance: ", variance.shape
    zero_rows = []
    for i in range(len(variance)):
        if np.all(variance[i]==0):
            zero_rows.append(i)
    new_mean = np.delete(mean, zero_rows, 0)    
    _variance = np.delete(variance, zero_rows, 0)
    new_variance = np.delete(_variance, zero_rows, 1)
    
    ind_mean_dict = {}
    for ind in zero_rows:
        ind_mean_dict[ind] = mean[ind]
        # print "New variance: ", new_variance
    return (new_mean, new_variance, ind_mean_dict)

############# Multivariate normal density ########
def multinormalpdf((mu, Sigma, ind_mean_dict), x):
        #print "Sigma: ", Sigma
        
        for ind in ind_mean_dict.keys():
                if np.all((x[ind] - ind_mean_dict[ind]) == 0):
                        return 0
        new_x = np.delete(x, ind_mean_dict.keys(), 0)
        denom = np.sqrt(np.linalg.det(Sigma))
        prob = np.exp(-0.5*(new_x-mu).T.dot(np.linalg.inv(Sigma)).dot(new_x-mu))/denom
        return prob


############# Classifier #########################
def generative(gausses,image):
        """
                given the image, return label with max prob
        """
        p_vector = np.array([(multinormalpdf(gausses[i],image)) for i in range(10)])
        # print p_vector
        return np.argmax(p_vector)
        
############# Execution: Train data using img_train ##########################
gausses = []
batches = []
for i in range(10):
    """
        fit gaussian model for each 0<=i<=9, store in gausses
    """
    batch = np.array([img_train[j] for j in range(40000) if lbl_train[j]==i])
    batches.append(batch.shape[0])
    gausses.append(gaussian_fit(batch))
f = open('y2.out','w')
print >> f, "11.2%"
for i in range(5000):
    print >> f, generative(gausses, img_test[i])
f.close()
