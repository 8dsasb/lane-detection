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


def laneRegion(img):
    image_height = img.shape[0]
    polygons = np.array(
        [[(170, image_height), (750, 170), (1060, image_height)]])
    image_mask = np.zeros_like(img)
    cv2.fillPoly(image_mask, polygons, 255)
    return image_mask


def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


# ****** EDGE DETECTION ******#


gray_image = grayScaleConv(img)
blurred_image = gaussianBlur(gray_image)
edge_image = cannyEdge(blurred_image)
image_mask = laneRegion(edge_image)
showImage(image_mask)
