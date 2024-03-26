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
    #coordinates for the polygon, in this case a triangle since it has three points
    polygons = np.array(
        [[(170, image_height), (750, 170), (1060, image_height)]])
    image_mask = np.zeros_like(img)
    #filling the mask with the polygon, the result is a black image with white triangle in the specified coordinates
    cv2.fillPoly(image_mask, polygons, 255)
    #performing bitwise and which returns a value if both of the pixel values being operated have the value 1, i.e 1 bitwiseand 1 = 1, else 0
    cropped_image = cv2.bitwise_and(img, image_mask)
    return cropped_image


#function to display detected lines in a black image
def display_lines(img, lines):
    background = np.zeros_like(img)
    for x1, y1, x2, y2 in lines:
        #reshaping only when we are using the output of the hough transformation which is a 3d image. however, if we use the avreaged image, then it is already a 1-d array so this line is being commented out
        # x1, y1, x2, y2 = i.reshape(4)
    
        cv2.line(background, (x1,y1), (x2,y2), (0,255,0), 10)
    
    return background

#hough transformation, for seeing if a series of points in the image belong to a line or not

def houghTransform(img):
    #image, row, theta, threshold
    image = cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
    return image

def lineOverImage(img, line_img):
    #displaying the detected line over the original image
    #the original image gets lets weight so that the detected line is much more prominent. 0.8 and 1
    line_over_image = cv2.addWeighted(img, 0.8, line_img, 1, 1)
    return line_over_image

def average_slope_intercept(img, lines):
    #arrays to store all the lines detected on the left and right side of the lane
    left_lines = []
    right_lines = []
    for i in lines:
        x1,y1,x2,y2 = i.reshape(4)
    #getting the slope and intercepts of the lines using polyfit
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        #the returned value is in the form of [slope,intercept]
        slope, intercept = parameters[0], parameters[1]
        #since the slope of the left lane line is negative, that is as we move positively down the x axis, the value in the y axis decreases and vice versa. it is the opposite for the right lane. 
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    #since the appeneded lines are an array of slope and intercept value, with the first element of each array being the slope and the second element of each array being the intercept, we can find the average of these values in the 0 axis, which is, selecting every elements of a low and summingn it vertically. this will give a final array with an average value of all the slopes and intercepts for left and right side (single lines for each side).
    left_lines_avg = np.average(left_lines, axis = 0)
    right_lines_avg = np.average(right_lines, axis = 0)
    # print(left_lines_avg, "left")
    # print(right_lines_avg, "riight")
    left_line = draw_lines(img, left_lines_avg)
    right_line = draw_lines(img, right_lines_avg) 
    return np.array([left_line, right_line])

def draw_lines(image, line):
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)

# def showVideo(video):
#     vid_cap = cv2.VideoCapture(video)
#     while (vid_cap.isOpened()):
#         _, current_frame = vid_cap.read()

