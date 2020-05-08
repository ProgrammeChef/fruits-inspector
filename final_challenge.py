#!/usr/bin/env python
"""
Final challenge implementation file.
"""

import cv2
import utils
import numpy as np

__author__ = "Marco Rossini"
__copyright__ = "Copyright 2020, Marco Rossini"
__date__ = "2020/05"
__license__ = "MIT"
__version__ = "1.0"

# ----------------------------------------------------------------------------------------------------------------------

def run():
    # Read and store all images into an array
    path = "./images/final_challenge"
    bw_images, bw_file_names, color_images, color_file_names = utils.get_images_as_array(path)

    # Iterate over all images
    for i in range(len(bw_images)):
        bw_image = bw_images[i]
        bw_file_name = bw_file_names[i]
        color_image = color_images[i]
        color_file_name = color_file_names[i]

        # Show current image and print its name
        utils.show_image(color_file_name, color_image)

        # Convert image to grayscale
        gray = cv2.cvtColor(bw_image, cv2.COLOR_RGB2GRAY)

        # Equalise histogram to improve contrast
        equalised = cv2.equalizeHist(gray)

        # Calculate optimal threshold as 'mode + factor * median / 2' (customizable)
        optimal = utils.get_optimal_threshold(equalised, 2.3)

        # Binarize the image to separate foreground and background
        threshold, binarized = cv2.threshold(equalised, optimal, 255, cv2.THRESH_BINARY)

        # Get fruit mask (biggest component)
        mask = utils.get_biggest_component(binarized)

        # Fill the holes
        filled = utils.fill_holes(mask)

        # Separate eventually touching objects by cutting out convexity defects
        # specifying a timeout of 1 iteration (customisable)
        separated = utils.separate_touching_objects(filled, 1, 1.35)

        # Get fruit mask after separation (biggest component)
        mask = utils.get_biggest_component(separated)

        # Apply a median blur to smooth mask
        blurred = utils.median_blur(mask, 3, 5)

        # Get grayscale fruit from filled mask
        fruit = cv2.bitwise_and(bw_image, bw_image, mask=blurred)

        # Apply a bilateral blur to remove noise but preserving edges
        fruit_blurred = cv2.bilateralFilter(fruit, 11, 100, 75)

        # Perform a Canny edge detection
        canny = cv2.Canny(fruit_blurred, 10, 95)

        # Get background mask by inverting fruit mask
        background = 255 - blurred

        # Dilate background mask to cut out the external edge
        kernel = np.ones((5, 5), np.uint8)
        background_dilated = cv2.dilate(background, kernel, iterations=3)

        # Remove external fruit contour
        defects = cv2.subtract(canny, background_dilated)

        # Apply a closing operation to consolidate detected edges
        structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        closed = cv2.morphologyEx(defects, cv2.MORPH_CLOSE, structuringElement)

        # Perform a connected components labeling to detect defects
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 4)

        # Get a copy of the original image for visualisation purposes
        display = color_image.copy()

        # Outline the fruit using its binary mask
        utils.draw_fruit_outline(display, blurred, 1)

        # Declare a defects counter (for visualisation purposes)
        defects_counter = 0

        # Iterate over the detected components to isolate and show defects
        for j in range(1, retval):
            # Isolate current binarized component
            component = utils.get_component(labels, j)
            defects_counter = utils.draw_defect(display, component, 2, 1.3, 0, float("inf"), 5)

        # Show processed image
        display_bw = closed.copy()
        utils.draw_fruit_outline(display_bw, blurred, 1, (255, 255, 255))
        utils.show_image(bw_file_name, display_bw)

        # Print detected defects number
        print(color_file_name + ": detected " + str(defects_counter) + " defect(s)")

        # Show original image highlighting defects
        utils.show_image(color_file_name, display)
