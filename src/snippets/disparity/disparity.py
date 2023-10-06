
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

directory = os.getcwd()

imgL = cv.imread(os.path.join(directory, "src/snippets/disparity", "left-0031.png"), cv.IMREAD_GRAYSCALE)
imgR = cv.imread(os.path.join(directory, "src/snippets/disparity", "right-0031.png"), cv.IMREAD_GRAYSCALE)

plt.imshow(imgL,'gray')
plt.imshow(imgR,'gray')
plt.show()

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()