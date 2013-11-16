# Final Project
import numpy as np
import matplotlib.pyplot as plt
import time

t = time.time()

##################### Import from digit_recognition_1
# def img_plot(image):
#     arr = np.asarray(image)
#     plt.imshow(arr, cmap = cm.Greys_r)
#     plt.show()
#     return 0
    
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
#     print "Predict: ", vote.index(max(vote))
#     print "Real: ", x[1]
    return vote.index(max(vote)), x[1] #return the ones most voted for

######################################################
# def normalize((image, label)):
#     """
#         reduce each dimension by 4 -- take average of a 4 by 4 matrix and make it a cell
#         in the 7 by 7 matrix
#     """
#     new = np.zeros((7,7))
#     index = 0
#     for i in range(0, 28, 4):
#         for j in range(0, 28, 4):
#             new[index/7][index % 7] += float(np.sum(image[i:(i+3), j:(j+3)]))/16
#             index += 1
#     return (new, label)

def normalize((image,label)):
    """
        reduce dimension to 7x7 by taking average of 
        overlapping 7x7 squares

    """
    new = np.zeros((7,7))
    index = 0
    for i in [0, 4, 7, 10, 14, 17, 21]:
        for j in [0, 4, 7, 10, 14, 17, 21]:
            new[index / 7][index % 7] += float(np.sum(image[i:(i+6),j:(j+6)]))/49
            index += 1
    return (new,label)

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
f_img_A = "C:/Users/Anh Huynh/Dropbox/Mathematics/Work/Fall_2013_Courses/trainingImagesA.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
f_img_B = "C:/Users/Anh Huynh/Dropbox/Mathematics/Work/Fall_2013_Courses/trainingImagesB.npy"
img_B = np.array(np.load(f_img_B, mmap_mode='r'))


f_lbl_A = "C:/Users/Anh Huynh/Dropbox/Mathematics/Work/Fall_2013_Courses/trainingLabelsA.npy"
lbl_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
A_set = zip(img_A,lbl_A)
# w_A_set = np.array([ np.append(normalize(A_set[np.random.randint(len(A_set))]), [1]) for i in range(5000)])
w_A_set = np.array([ np.append(normalize(A_set[i]), [1]) for i in range(20000)])
f_lbl_B = "C:/Users/Anh Huynh/Dropbox/Mathematics/Work/Fall_2013_Courses/trainingLabelsB.npy"
lbl_B = np.array(np.load(f_lbl_B, mmap_mode='r'))
B_set = zip(img_B,lbl_B)
# w_B_set = np.array([ np.append(normalize(B_set[np.random.randint(len(B_set))]), [1]) for i in range(5000)])
w_B_set = np.array([ np.append(normalize(B_set[i]), [1]) for i in range(20000)])

loss = 0
for i in range(20000):
    predict, real = KNN(3, w_A_set, w_B_set[i])
#     plt.imshow(w_B_set[i][0])
#     plt.show()
    if predict != real:
        loss += 1
print "Loss: ", loss/20000.0, "%"
print time.time()-t