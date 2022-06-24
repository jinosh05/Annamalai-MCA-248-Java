import numpy as np
import cv2
import os
cwd = os.getcwd()

img = cv2.imread(os.path.join(cwd, 'nature.png'))
Z = img.reshape((-1, 3))
# convert to np.float32 Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
# ret,label,center=cv2.kmeans(Z,K, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
_, label, center = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original images

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# segmentation
gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
ret, threshseg = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(cwd, 'img_CV2_95.jpg'), threshseg)
cv2.imwrite(os.path.join(cwd, 'img_CV2_94.jpg'), res2)
cv2.imshow('threshseg', threshseg)
cv2.imshow('thresh', thresh)
cv2.imshow('res2', res2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
