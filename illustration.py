"""
	This file produces the plots needed for the report and poster.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import time

t = time.time()
"""
########################import training images and labels#########################
"""
# f_img_A = "trainingImagesA.npy"
# img_A = np.array(np.load(f_img_A, mmap_mode='r'))
# f_img_B = "trainingImagesB.npy"
# img_B = np.array(np.load(f_img_B, mmap_mode='r'))
# f_img_A_red = "trainingImagesA_reduced.npy"
# img_A_red = np.array(np.load(f_img_A_red, mmap_mode='r'))
# f_img_B_red = "trainingImagesB_reduced.npy"
# img_B_red = np.array(np.load(f_img_B_red, mmap_mode='r'))
# f_lbl_A = "trainingLabelsA.npy"
# label_A = np.array(np.load(f_lbl_A, mmap_mode='r'))
# f_lbl_B = "trainingLabelsB.npy"
# label_B = np.array(np.load(f_lbl_B, mmap_mode='r'))

"""
########################plot a few sample images###################################
"""
# for i in range(60):
# 	"""
# 		to get distinct images for 10 digits
# 	"""
# 	plt.imshow(img_A[i],cmap = cm.Greys_r)
# 	plt.show()
# for i in range(60):
# 	arr = img_A_red[i].reshape(7,7)
# 	plt.imshow(arr, cmap = cm.Greys_r)
# 	plt.show()
# arr = img_B_red[0].reshape(7,7)
# plt.imshow(arr, cmap = cm.Greys_r)
# plt.show()
k=[500,1000,5000,10000, 20000]
# l500 = [17.36, 17.34, 18.44, 19.56]
# l1000 = [14.36, 12.6, 13.76, 14.78]
# l5000 = [9.86, 8.94, 10.00, 9.22]
# l10000 = [7.50, 7.28, 8.08, 8.66]
# l500 = [85.76, 82.50, 80.60, 81.2]
# l1000 = [81.10, 84.92, 82.84, 84.42]
# l5000 = [77.20, 79.58, 78.20, 80.12]
# l10000 = [77.14, 75.6, 77.82, 78.42]
loss = [83.98, 64.06, 26.64, 23.94, 16.82]
plt.xlabel("Training size")
plt.ylabel("Validation loss (%)")
plt.title("Graphs of Validation Losses for GDA on nonreduced images")
plt.plot(k, loss, "r-")
# plt.plot(k, l1000, "b-", label = 'size = 1000')
# plt.plot(k, l5000, "g-", label = 'size = 5000')
# plt.plot(k, l10000, "k-", label = 'size = 10000')
# plt.legend()
plt.show()