import cv2
import utils
import numpy as np
from scipy.spatial.distance import cdist


def run_clustering():
    print("Method 1: using a clustering algorithm (K-means).")

    # Read and store all images into an array
    path = "./images/second_task"
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

        # Calculate optimal threshold as 'mode + median / 2'
        optimal = utils.get_optimal_threshold(gray)

        # Binarize the image to separate foreground and background
        threshold, binarized = cv2.threshold(gray, optimal, 255, cv2.THRESH_BINARY)

        # Get fruit mask (biggest component)
        biggest_component = utils.get_biggest_component(binarized)

        # Fill the holes
        filled = utils.fill_holes(biggest_component)

        # Get a copy of the original image for visualisation purposes
        display = color_image.copy()

        # Outline the fruit using the binary mask
        utils.outline_fruit(display, filled, 1)

        # Erode one time to remove dark contour
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(filled, kernel, iterations=1)

        # Apply a median blur to remove noise but preserving edges
        blurred = utils.median_blur(color_image, 3, 3)

        # Get fruit from filled mask
        fruit = cv2.bitwise_and(blurred, blurred, mask=eroded)

        # Convert isolated fruit to Lab color space to preserve perceptual meaning
        fruit_lab = cv2.cvtColor(fruit, cv2.COLOR_BGR2LAB)

        # Detect dominant colors performing a K-means clustering (with K=3) on 'a' and 'b' channels of the Lab image
        # (L is excluded to be robust to lighting's variations)
        average_labs, colors, labels = utils.get_dominant_colors(fruit_lab, 3)

        # Discriminate russet color among detected ones measuring distance from 'dark brown'
        russet_index = utils.get_russet_index(average_labs)
        russet_sample = average_labs[russet_index]

        # Show a sample of the detected color
        utils.show_sample_lab(russet_sample, "Detected russet sample", 200, 200)

        # Get isolated russet on fruit as the corresponding cluster of pixels
        russet_component = utils.get_component(labels, russet_index)
        russet = cv2.bitwise_and(fruit, fruit, mask=russet_component)

        # Compute mask area to set an upper bound for drawing a defect
        eroded_area = cv2.countNonZero(eroded)

        # Show defect on original image
        defects = utils.draw_defect(display, russet_component, 2, 1.2, 15, 4 * eroded_area, 5)

        # Print detected defects number
        print(color_file_name + ": detected " + str(defects) + " defect(s)")

        # Show isolated russet
        # utils.show_image(color_file_name, russet)

        # Show original image highlighting defects
        utils.show_image(color_file_name, display)


def run_samples():
    print("\nMethod 2: using samples and Malahanobis distance.")

    # Read and store all images into an array
    path = "./images/second_task"
    samples_path = "./images/second_task/samples"
    bw_images, bw_file_names, color_images, color_file_names = utils.get_images_as_array(path)
    samples, samples_file_names = utils.get_samples_as_array(samples_path)

    # Iterate over all images
    for i in range(len(bw_images)):
        bw_image = bw_images[i]
        bw_file_name = bw_file_names[i]
        color_image = color_images[i]
        color_file_name = color_file_names[i]

        # Show current image and print its name
        utils.show_image(color_file_name, color_image, 0, 0)

        # Convert image to grayscale
        gray = cv2.cvtColor(bw_image, cv2.COLOR_RGB2GRAY)

        # Calculate optimal threshold as 'mode + median / 2'
        optimal = utils.get_optimal_threshold(gray)

        # Binarize the image to separate foreground and background
        threshold, binarized = cv2.threshold(gray, optimal, 255, cv2.THRESH_BINARY)

        # Get fruit mask (biggest component)
        biggest_component = utils.get_biggest_component(binarized)

        # Fill the holes
        filled = utils.fill_holes(biggest_component)

        # Get a copy of the original image for visualisation purposes
        display = color_image.copy()

        # Outline the fruit using the binary mask
        utils.outline_fruit(display, filled, 1)

        # Erode one time to remove dark contour
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(filled, kernel, iterations=1)

        # Apply a median blur to remove noise but preserving edges
        blurred = utils.median_blur(color_image, 3, 3)

        # Get fruit from filled mask
        fruit = cv2.bitwise_and(blurred, blurred, mask=eroded)

        # Convert isolated fruit to Lab color space to preserve perceptual meaning
        fruit_lab = cv2.cvtColor(fruit, cv2.COLOR_BGR2LAB)

        # Create data structures to store total covariance and mean of samples
        covariance_tot = np.zeros((3, 3), dtype="float64")
        mean_tot = np.zeros((1, 3), dtype="float64")

        # Iterate over samples to compute the reference color (i.e. mean of samples) and its total covariance
        for s in samples:
            s_lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
            s_lab_r = s_lab.reshape(s_lab.shape[0] * s_lab.shape[1], 3)
            cov, mean = cv2.calcCovarMatrix(s_lab_r, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
            covariance_tot = np.add(covariance_tot, cov)
            mean_tot = np.add(mean_tot, mean)

        # Compute the mean (reference color) as the mean of the means of all the samples
        mean_avg = mean_tot / len(samples)

        # Compute the inverse of the covariance matrix (needed to measure Mahalanobis distance)
        inv_cov = cv2.invert(covariance_tot, cv2.DECOMP_SVD)[1]

        russet_component = np.zeros_like(binarized)

        # Compute pixel-wise Mahalanobis distance between fruit and reference color
        for r in range(fruit_lab.shape[0]):
            for c in range(fruit_lab.shape[1]):
                # Compute the distance only for fruit's pixels (excluding background)
                if filled[r][c]:
                    # Get the pixel as a numpy array (needed for cdist)
                    p = np.array(fruit_lab[r][c]).reshape(1, 3)
                    # Compute pixel-wise Mahalanobis distance
                    dist = cdist(p, mean_avg, 'mahalanobis', VI=inv_cov)
                    # If distance is small, 'p' is a russet's pixel
                    if dist < 1.9:
                        # Store russet's pixel location
                        russet_component[r][c] = 255

        # Get isolated russet on fruit as the corresponding cluster of pixels
        russet = cv2.bitwise_and(fruit, fruit, mask=russet_component)

        # Compute mask area to set an upper bound for drawing a defect
        eroded_area = cv2.countNonZero(eroded)

        # Show defect on original image
        defects = utils.draw_defect(display, russet_component, 2, 1.2, 55, 4 * eroded_area, 5)

        # Print detected defects number
        print(color_file_name + ": detected " + str(defects) + " defect(s)")

        # Show isolated russet
        # utils.show_image(color_file_name, russet)

        # Show original image highlighting defects
        utils.show_image(color_file_name, display)
