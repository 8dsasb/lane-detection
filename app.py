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
    cropped_image = cv2.bitwise_and(img, image_mask)
    return cropped_image


def display_lines(img, lines):
    background = np.zeros_like(img)
    for i in lines:
        x1, y1, x2, y2 = i.reshape(4)
        cv2.line(background, (x1,y1), (x2,y2), (0,255,0), 10)
    return background

#hough transformation

def houghTransform(img):
    #image, row, theta, threshold
    image = cv2.HoughLinesP(img, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
    return image

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


