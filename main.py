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


# Calculate the average slope and intercept of detected lines
averaged_lines = ap.average_slope_intercept(image, hough_trans_image)

# Display the averaged lines
line_only_image = ap.display_lines(image, averaged_lines)

line_over_image = ap.line_on_image(image, line_only_image)

list_of_images = {"Gray Img": gray_image, "Blurred Img":blurred_image, "Canny Edge": edge_image, "Cropped Img": cropped_image, "Lines Only": line_only_image, "Detected lines": line_on_image, "Averaged Line": line_over_image}

#Show the final images with lines

ap.show_imgs_grid(list_of_images)

# Lane detection in a video

cap = cv2.VideoCapture("lane_video.mp4")  # Open video file
while(cap.isOpened()):
    ret, frame = cap.read()  # Read a frame from the video
    
    if not ret:
        break  # Break if no frame is returned (end of video)

    # Edge detection steps
    gray_image = ap.grayScaleConv(frame)  # Convert to grayscale
    blurred_image = ap.gaussianBlur(gray_image)  # Apply Gaussian blur
    edge_image = ap.cannyEdge(blurred_image)  # Detect edges using Canny
    cropped_image = ap.laneRegion(edge_image)  # Crop region of interest

    # Hough Transform to detect lines
    hough_trans_image = ap.houghTransform(cropped_image)

    # Calculate the average slope and intercept for detected lines
    averaged_lines = ap.average_slope_intercept(frame, hough_trans_image)

    # Display the lines on the frame
    line_only_image = ap.display_lines(frame, averaged_lines)

    line_over_image = ap.line_on_image(frame, line_only_image)

    # Show the resulting image with lines overlaid
    cv2.imshow("Lane Detection", line_over_image)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()