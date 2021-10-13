import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time

im = cv2.imread("grayscale_laser.png", cv2.IMREAD_GRAYSCALE)

#plt.imshow(im)
#plt.show()
print(im.shape)

start = time.time()
top_indices = np.argpartition(im, -5, axis=0)[-5:]
top_values = np.partition(im, -5, axis=0)[-5:]

h = top_values * top_indices

numerator = np.sum(h, axis=0)
denominator = np.sum(top_values, axis=0)
center = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0, casting='unsafe')
xs = np.arange(0,im.shape[1], 1)
plt.scatter(xs, center, s=1, color='g')
plt.xlim(0,im.shape[1])
plt.ylim(im.shape[0], 0)
plt.show()
print(f'execution time:', time.time() - start)