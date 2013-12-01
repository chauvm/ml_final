"""
Final Project: k-nearest-neighbor
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
t = time.time()  
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
    Predicts using KNN trained with training_set
    """
    neighbors = find_neighbors(k, x, train_set)
    vote = [0 for i in range(10)]
    for neighbor in neighbors:
        vote[neighbor[1]] += 1
    return vote.index(max(vote)) #return the ones most voted for

#######################################################
f_img_A = "trainingImagesA_reduced.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
f_img_B = "trainingImagesB_reduced.npy"
img_B = np.array(np.load(f_img_B, mmap_mode='r'))
img_train = np.append(img_A,img_B,axis=0)
f_test = "testingImages_red.npy"
img_test = np.array(np.load(f_test, mmap_mode='r'))

f_lbl_A = "trainingLabelsA.npy"
lbl_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
f_lbl_B = "trainingLabelsB.npy"
lbl_B = np.array(np.load(f_lbl_B, mmap_mode='r'))
lbl_train = np.append(lbl_A,lbl_B, axis=0)
train_set = zip(img_train, lbl_train)

f = open('y1.out','w')
print >> f, "5.15%"

for i in range(5000):
    predict = KNN(5, train_set, img_test[i])
    print >> f, predict 
    if i%500==0:
        print i, time.time()-t

f.close()