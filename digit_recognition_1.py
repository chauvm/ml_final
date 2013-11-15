"""
k-nearest neighbors for digit recognition.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image

t = time.time()
k = 3
#############Read train_set and test_set##################################
f_img_A = "trainingImagesA.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
f_img_B = "trainingImagesB.npy"
img_B = np.array(np.load(f_img_B, mmap_mode='r'))
f_lbl_A = "trainingLabelsA.npy"
lbl_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
A_set = zip(img_A,lbl_A)
w_A_set = np.array([ np.append(A_set[np.random.randint(len(A_set))], [1]) for i in range(5000)])
# w_A_set = np.array([ np.append(point, [1]) for point in A_set])
f_lbl_B = "trainingLabelsB.npy"
lbl_B = np.array(np.load(f_lbl_B, mmap_mode='r'))
B_set = zip(img_B,lbl_B)
sub_B_set = [B_set[np.random.randint(len(B_set))] for i in range(100)]

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
    add the weight into the training set, so xy[2]
    """
    'Predicts using KNN trained with training_set'
    neighbor = find_neighbors(k, x[0], train_set)
    vote = [0 for i in range(10)]
    for xy in neighbor:
        vote[xy[1]]+=xy[2]
    return vote.index(max(vote)) #return the ones most voted for

def KNN_train(k,train_set,point):
    neighbor = find_neighbors(k, point[0], train_set)
    for xy in neighbor:
        if xy[1] == point[1]:
            xy[2]+=1
    return time.time()-t #return the ones most voted for


def loss():
    return sum([1 for point in sub_B_set if point[1]!=KNN(k,w_A_set,point[0])])
    # return sum([1 for point in small_test if point[1]!=KNN(k,train_set,point[0])])

for point in w_A_set:
    KNN_train(k,w_A_set,point)
print time.time()-t
