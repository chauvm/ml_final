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
    neighbors = find_neighbors(k, x[0], train_set)
    vote = [0 for i in range(10)]
    for neighbor in neighbors:
        vote[neighbor[1]] += 1
    return vote.index(max(vote)), x[1] #return the ones most voted for

#######################################################
f_img_A = "trainingImagesA.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
f_img_B = "trainingImagesB.npy"
img_B = np.array(np.load(f_img_B, mmap_mode='r'))

f_lbl_A = "trainingLabelsA.npy"
lbl_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
f_lbl_B = "trainingLabelsB.npy"
lbl_B = np.array(np.load(f_lbl_B, mmap_mode='r'))

def demo(K, train_sizes):
	losses = np.array([0]*(len(K))*len(train_sizes)).reshape(len(K),len(train_sizes))
	for size in train_sizes:
		for k in K:
			print "doing k = ", k, ", and size = ", size
			train_indices = np.array([np.random.randint(len(img_A)) for i in range(size)])
			img_train = np.array([img_A[i] for i in train_indices])
			lbl_train = np.array([lbl_A[i] for i in train_indices])
			train_set = zip(img_train,lbl_train)
			loss = 0
			for i in range(5000):
				j = np.random.randint(len(img_B))
				predict, real = KNN(k,train_set,[img_B[j], lbl_B[j]])
				if predict != real:
					losses[K.index(k)][train_sizes.index(size)] += 1
					loss += 1
			losses[K.index(k)][train_sizes.index(size)]= losses[K.index(k)][train_sizes.index(size)]/5000.0
			loss/= 5000.0
			# print losses[K.index(k)][train_sizes.index(size)]
			print loss
			print time.time()-t
	return losses

Losses = demo([3,5,7,11],[500,1000,5000,10000])


"""
losses for reduced images
size 500: 17.36%, 17.34%, 18.44%, 19.56%
size 1000: 14.36%, 12.6%, 13.76%, 14.78%
size 5000: 9.86%, 8.94%, 10.00%, 9.22%
size 10000: 7.50%, 7.28%, 8.08%, 8.66%

losses for nonreduced images  
size 500: 85.76%, 82.50%, 80.60%, 81.2%
size 1000: 81.10%, 84.92%, 82.84%, 84.42%
size 5000: 77.20%, 79.58%, 78.20%, 80.12%
size 10000: 77.14%, 75.6%, 77.82%, 78.42%
"""