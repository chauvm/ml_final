# Final Project
import numpy as np
size_data = 20000
def normalize((image)):
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
    return new

f_img_A = "trainingImagesA.npy"
img_A = np.array(np.load(f_img_A, mmap_mode='r'))
f_img_B = "trainingImagesB.npy"
img_B = np.array(np.load(f_img_B, mmap_mode='r'))

img_red_A = np.array([ normalize(img_A[i]) for i in range(size_data)])
img_red_B = np.array([ normalize(img_B[i]) for i in range(size_data)])
np.save("trainingImagesA_reduced.npy",img_red_A)
np.save("trainingImagesB_reduced.npy",img_red_B)
