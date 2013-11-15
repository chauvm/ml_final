# Final Project
import numpy as np
import matplotlib.pyplot as plt

##################### Import from digit_recognition_1
def img_plot(image):
    arr = np.asarray(image)
    plt.imshow(arr, cmap = cm.Greys_r)
    plt.show()
    return 0
    
def dist(x1, x2):
    'Computes the square of Euclidean distance between two points'
    return np.linalg.norm((x1-x2))

def find_neighbors(k, x, train_set):
    'Finds k nearest neighbors to x from train_set'
    if len(train_set) <= k:
        return train_set
    else:
        neighbors = [[train_set[i], dist(train_set[i][0], x)] for i in range(k)]
        for j in range(k, len(train_set)):
            distances = zip(*neighbors)[1]
            currentMax = max(distances)
            indexMax = distances.index(currentMax)
            curDistance = dist(train_set[j][0], x)
            if curDistance < currentMax:
                del neighbors[indexMax]
                neighbors.append([train_set[j], curDistance])
        return zip(*neighbors)[0]

def KNN(k, train_set, x):
    """
    add the weight into the training set, so neighbor[2]
    """
    'Predicts using KNN trained with training_set'
    neighbors = find_neighbors(k, x[0], train_set)
    vote = [0 for i in range(10)]
    for neighbor in neighbors:
        vote[neighbor[1]] += 1
##    print "Predict: ", vote.index(max(vote))
##    print "Real: ", x[1]
    return vote.index(max(vote)), x[1] #return the ones most voted for

######################################################
def normalize((image, label)):
    """
        reduce each dimension by 4 -- take average of a 4 by 4 matrix and make it a cell
        in the 7 by 7 matrix
    """
    new = np.zeros((7,7))
    index = 0
    for i in range(0, 28, 4):
        for j in range(0, 28, 4):
            new[index/7][index % 7] += float(np.sum(image[i:(i+3), j:(j+3)]))/16
            index += 1
    return (new, label)
def clusterize(number, X, labels):
    indices = []
    for i, num in enumerate(labels):
        if num == number:
            indices.append(i)
    newImages = []
    for i in indices:
        newImages.append(normalize(X[i]))
    return newImages

def findMean(cluster):
    return sum(cluster)/len(cluster)

def distance(image1, image2):
    return np.sum(np.absolute(image1 - image2))


#######################################################

# Load data
##X = np.load('trainingImagesA.npy', mmap_mode='r')
##labels = np.load('trainingLabelsA.npy', mmap_mode='r')

f_img_A = "trainingImagesA.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))



f_lbl_A = "trainingLabelsA.npy"
lbl_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
A_set = zip(img_A,lbl_A)
w_A_set = np.array([ np.append(normalize(A_set[np.random.randint(len(A_set))]), [1]) for i in range(5000)])

loss = 0
##for i in range(500):
##    predict, real = KNN(3, w_A_set, w_A_set[i])
##    if predict != real:
##        loss += 1
##print "Loss: ", loss
##clusters = []
##means = []
##for num in range(1, 10):
##    cluster = clusterize(num, X, labels)
##    clusters.append(cluster)
##    means.append(findMean(cluster))
##
##distances_ones = []
##first_one = clusters[0][0]
##for m in means:
##    distances_ones.append(distance(first_one, m))
##print distances_ones
plt.imshow(img_A[0])
plt.show()
