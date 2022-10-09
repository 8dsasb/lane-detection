# ***** IMPORTS ****
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ***** VARIABLES INITIALIZATION *****

original_img = cv2.imread('F:/proj/laneDetection/lane.jpg')
img = np.copy(original_img)

# ***** FUNCTIONS *****


def grayScaleConv(img):
    return (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))


def gaussianBlur(img):
    return (cv2.GaussianBlur(img, (5, 5), 0))


def cannyEdge(img):
    # threshold low:high = 1:3
    return (cv2.Canny(img, 40, 120))


def showImage(img):
    plt.imshow('Image', img)
    plt.show()


# ****** EDGE DETECTION ******#


gray_image = grayScaleConv(img)
blurred_image = gaussianBlur(gray_image)
edge_image = cannyEdge(blurred_image)
showImage(edge_image)
