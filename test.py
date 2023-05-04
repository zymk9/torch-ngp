import numpy as np
import cv2


a = np.load('train_d.npy')
b = np.load('test_d.npy')

print((a-b).sum())