import app as ap
import cv2
import numpy as np

# ****** INITIALIZATIONS *****
original_img = cv2.imread('lane.jpg')
image = np.copy(original_img)

# ****** EDGE DETECTION ******
gray_image = ap.grayScaleConv(image)  # Convert image to grayscale
blurred_image = ap.gaussianBlur(gray_image)  # Apply Gaussian blur
edge_image = ap.cannyEdge(blurred_image)  # Detect edges using Canny
cropped_image = ap.laneRegion(edge_image)  # Crop region of interest
hough_trans_image = ap.houghTransform(cropped_image)  # Detect lines using Hough Transform

# Display detected lines before averaging
line_only_image = ap.display_lines(image, hough_trans_image)
line_on_image = ap.line_on_image(image, line_only_image)

list_of_images = [gray_image, blurred_image, edge_image, cropped_image, line_only_image, line_on_image]
# Calculate the average slope and intercept of detected lines
averaged_lines = ap.average_slope_intercept(image, hough_trans_image)

# Display the averaged lines
# line_only_image = ap.display_lines(image, averaged_lines)

# line_over_image = ap.lineOverImage(image, line_only_image)

# Optional: Show the final image with lines
# ap.show_image(cropped_image)

# ap.show_image(line_over_image)
# ap.showMultiple(list_of_images)

ap.show_imgs_grid(list_of_images)