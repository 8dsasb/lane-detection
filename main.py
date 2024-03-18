import app as ap
import cv2
import numpy as np

# ****** INITIALIZATIONS *****#

original_img = cv2.imread('lane.jpg')
img = np.copy(original_img)


# ****** EDGE DETECTION ******#


gray_image = ap.grayScaleConv(img)
#not needed since canny itself applies blur to the image
blurred_image = ap.gaussianBlur(gray_image)
edge_image = ap.cannyEdge(blurred_image)
image_mask = ap.laneRegion(edge_image)

#show image commands for all variations of the image above

# #gray image
# ap.showImage(gray_image)
# #blurred image
# ap.showImage(blurred_image)
# #edge image
# ap.showImage(edge_image)


#image mask
ap.showImage(image_mask)