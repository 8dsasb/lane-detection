import app as ap
import cv2
import numpy as np

# ****** INITIALIZATIONS *****#

original_img = cv2.imread('lane.jpg')
image = np.copy(original_img)


# ****** EDGE DETECTION ******#


gray_image = ap.grayScaleConv(image)
#not needed since canny itself applies blur to the image
blurred_image = ap.gaussianBlur(gray_image)
edge_image = ap.cannyEdge(blurred_image)
cropped_image = ap.laneRegion(edge_image)
hough_trans_image = ap.houghTransform(cropped_image)
line_over_image = ap.display_lines(image, hough_trans_image)

#show image commands for all variations of the image above

# #gray image
# ap.showImage(gray_image)
# #blurred image
# ap.showImage(blurred_image)
# #edge image
# ap.showImage(edge_image)

#cropped_image
# ap.showImage(cropped_image)

# #hough image
# ap.showImage(hough_trans_image)

#display lines
ap.showImage(line_over_image)