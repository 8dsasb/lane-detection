# ***** IMPORTS ****
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ***** FUNCTIONS *****

def grayScaleConv(img):
    """
    Convert an image to grayscale.

    Args:
        img (numpy.ndarray): The input image in RGB format.

    Returns:
        numpy.ndarray: The grayscale version of the input image.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussianBlur(img):
    """
    Apply Gaussian blur to an image to reduce noise and detail.

    Args:
        img (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The blurred image.

    Notes:
        - The kernel size is (5, 5), and standard deviation is 0 (optimal value).
    """
    return cv2.GaussianBlur(img, (5, 5), 0)

def cannyEdge(img):
    """
    Detect edges in an image using the Canny edge detection algorithm.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with detected edges.

    Notes:
        - Thresholds are set to 40 (low) and 120 (high), with a ratio of 1:3.
    """
    return cv2.Canny(img, 40, 120)

def laneRegion(img):
    """
    Applies a polygonal mask (triangle) to an input image, isolating a region of interest.

    Args:
        img (numpy.ndarray): The input image, typically in BGR format.

    Returns:
        numpy.ndarray: The image with the region inside the triangle retained and the rest blacked out.

    Notes:
        - Mask is defined as a triangle with vertices at (170, image_height), (750, 170), and (1060, image_height).
    """
    image_height = img.shape[0]
    polygons = np.array([[(170, image_height), (750, 170), (1060, image_height)]])
    image_mask = np.zeros_like(img)
    cv2.fillPoly(image_mask, polygons, 255)
    cropped_image = cv2.bitwise_and(img, image_mask)
    return cropped_image

def display_lines(img, lines):
    """
    Display detected lines on a black background.

    Args:
        img (numpy.ndarray): The input image (used only for size).
        lines (numpy.ndarray): Array of detected lines, each defined by two points (x1, y1, x2, y2).

    Returns:
        numpy.ndarray: A black image with the detected lines drawn in green.
    """
    background = np.zeros_like(img)
    # old error
    # for x1, y1, x2, y2 in lines:
    #     cv2.line(background, (x1, y1), (x2, y2), (0, 255, 0), 8)
    # new modified code:
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(background, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return background

def houghTransform(img):
    """
    Apply the Hough Transform to detect lines in the image.

    Args:
        img (numpy.ndarray): The input binary image where edges are detected.

    Returns:
        numpy.ndarray: Array of detected lines in the form [x1, y1, x2, y2].
    """
    return cv2.HoughLinesP(img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

def draw_lines(image, line):
    """
    Draw a line on the image given the slope and intercept.

    Args:
        image (numpy.ndarray): The input image.
        line (tuple): A tuple (slope, intercept) representing the line.

    Returns:
        numpy.ndarray: The coordinates of the line to be drawn.
    """
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    """
    Calculate the average slope and intercept for the left and right lane lines.

    Args:
        img (numpy.ndarray): The input image (used for drawing the lines).
        lines (numpy.ndarray): Array of detected lines.

    Returns:
        numpy.ndarray: The left and right lane lines drawn on the image.
    """
    left_lines = []
    right_lines = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0], parameters[1]

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_lines_avg = np.average(left_lines, axis=0)
    right_lines_avg = np.average(right_lines, axis=0)

    left_line = draw_lines(img, left_lines_avg)
    right_line = draw_lines(img, right_lines_avg)

    return np.array([left_line, right_line])

def line_on_image(img, line_only_img):
    line_on_image = cv2.addWeighted(img, 0.8, line_only_img, 1, 1)
    return line_on_image
    
def show_image(img):
    """
    Display an image in a window.

    Args:
        img (numpy.ndarray): The image to be displayed.
    """
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def show_imgs_grid(image_dict):
    # Determine the number of images
    labels = list(image_dict.keys())
    images = list(image_dict.values())

    num_images = len(images)
    
    # Calculate grid dimensions: This will give an approximate square grid
    cols = 3  # Define number of columns
    rows = math.ceil(num_images / cols)  # Calculate number of rows dynamically

    # Create subplots with dynamic grid size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes array to make it easier to iterate through
    axes = axes.flatten()

    # Loop through each image and corresponding axis
    for idx, img in enumerate(images):
        ax = axes[idx]  # Get the axis for the current image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
        ax.axis('off')  # Turn off axis labels

        # Labels below the image
        ax.set_title(labels[idx], fontsize = 10, pad = 10)
    # Turn off axes for any unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    # Adjust layout, avoid label merging and show the grid of images
    
    plt.show()

# def showMultiple(images, grid_shape = (1,3)):
#     """
#     Display multiple images together in a grid
    
#     Args:
#         images: Array of images to be displayed
#     """
#     # Determining the grid size
#     rows, cols = grid_shape
#     # Empty array to store resized images
#     resized_images = []

#     # Dimensions for resizing all of the images for uniformity
#     max_height = max([img.shape[0] for img in images])
#     max_width = max([img.shape[1] for img in images])


#     # Resizing the images
#     for img in images:
#         resized_img = cv2.resize(img, (max_width, max_height))
#         resized_images.append(resized_img)

#     # Arranging images in a grid
#     # Concatenating images horizontally for each row
#     rows_images = []
#     for i in range(rows):
#         row_images = resized_images[i * cols:(i+1) * cols]
#         if row_images:
#             row = cv2.hconcat(row_images)
#             rows_images.append(row)
    
#     # Concatenate all rows vertically
#     grid_image = cv2.vconcat(rows_images)

#     # Display the result
#     cv2.imshow("Image Grid", grid_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


    
