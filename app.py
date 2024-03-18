# ***** IMPORTS ****
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ***** VARIABLES INITIALIZATION *****


# ***** FUNCTIONS *****


def grayScaleConv(img):
    return (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

#optional step since canny applies blur
def gaussianBlur(img):
    #image, kernel size and standard deviation
    return (cv2.GaussianBlur(img, (5, 5), 0))


#derivative between adjacent pixels to spot the pixel intensity difference
#making it easier to spot the edges if there is a big difference (change in gradient)


def cannyEdge(img):
    # threshold low:high = 1:3
    return (cv2.Canny(img, 40, 120))


def laneRegion(img):
    #numpy tuple given by shape = (row,column) so row = height of the image 
    image_height = img.shape[0]
    polygons = np.array(
        [[(170, image_height), (750, 170), (1060, image_height)]])
    image_mask = np.zeros_like(img)
    cv2.fillPoly(image_mask, polygons, 255)
    return image_mask

#hough  transformation

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


