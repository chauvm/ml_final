# Final Project
import numpy as np
size_data = 5000
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

f_img = "testingImages.npy"
imgs = np.array(np.load(f_img, mmap_mode='r'))

img_red = np.array([ normalize(imgs[i]).flatten() for i in range(size_data)])
np.save("testingImages_red.npy",img_red)
