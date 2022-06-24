import cv2 as cv
import numpy as np
import os
cwd = os.getcwd()
image = cv.imread(os.path.join(cwd, 'nature.png'))
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
sharpened = cv.filter2D(image, -1, kernel)
# applying the sharpening kernel to the input image & displaying it.
cv.imshow('original Image', image)
cv.imshow('Image Sharpening', sharpened)
cv.waitKey(0)
cv.destroyAllWindows()
