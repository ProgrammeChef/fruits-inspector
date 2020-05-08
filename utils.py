#!/usr/bin/env python
"""
Utility functions file.
"""

import glob
import cv2
import math
import numpy as np
from scipy import stats
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

__author__ = "Marco Rossini"
__copyright__ = "Copyright 2020, Marco Rossini"
__date__ = "2020/05"
__license__ = "MIT"
__version__ = "1.0"

# ----------------------------------------------------------------------------------------------------------------------

# RGB shade of dark brown
dark_brown_rgb = [71, 56, 27]


# Processing functions

def median_blur(image, kernel_size, iterations):
    for i in range(iterations):
        image = cv2.medianBlur(image, kernel_size)

    return image


def separate_component(component, binarized, threshold):
    points = get_convexity_points(component)

    filtered_points = filter_points(binarized, points, 9, threshold)

    sorted_points = sort_points_pairwise(filtered_points)

    for p in sorted_points:
        # We impose a maximum distance threshold to avoid certainly wrong connections (customisable)
        if p[0] <= 12:
            cv2.line(binarized, p[1], p[2], (0, 0, 0), 2)

    return binarized


def filter_points(binarized, points, radius, tolerance):
    result = []

    for p in points:
        window = binarized[p[1] - radius:p[1] + radius, p[0] - radius:p[0] + radius]
        white_pixels = cv2.countNonZero(window)
        black_pixels = 2 * radius * 2 * radius - white_pixels

        if white_pixels / black_pixels > tolerance:
            result.append(p)

    return result


def sort_points_pairwise(points):
    sorted_points = []

    if len(points) > 0:
        distance_matrix = dist.cdist(np.array(points), np.array(points))

        for r in range(distance_matrix.shape[0]):
            distance_matrix[r][r] = 9999

        # We solve an assignment problem exploiting the Hungarian algorithm (also known as Kuhn-Munkres algorithm)
        rows, cols = linear_sum_assignment(distance_matrix)

        rows = rows[:len(rows) // 2]
        cols = cols[:len(cols) // 2]

        for i in range(len(rows)):
            temp = [distance_matrix[rows[i]][cols[i]], points[rows[i]], points[cols[i]]]
            sorted_points.append(temp)

    return sorted_points


def separate_touching_objects(binarized, timeout, threshold):
    while True:
        separated = 0
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, 4)

        for k in range(1, retval):
            # Isolate current binarized component
            component = get_component(labels, k)

            binarized = separate_component(component, binarized, threshold)
            separated += 1

        if separated == 0 or timeout == 0:
            break

        timeout -= 1

    return binarized


def filter_duplicated_contours(contours, min_parallax):
    measured = []

    for c in contours:
        if len(c) >= 5:
            centroid, _, _ = cv2.fitEllipse(c)
            area = cv2.contourArea(c)
            measured.append([c, centroid, area])

    filtered = []

    for m in measured:
        if is_not_duplicate(m, measured, min_parallax):
            filtered.append(m[0])

    return filtered


def rgb_to_lab(rgb):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = rgb
    result = cv2.cvtColor(temp, cv2.COLOR_RGB2LAB)[0][0]

    return result


# Boolean functions

def is_not_duplicate(m, measured, min_parallax):
    for c in measured:
        if np.array_equal(m[0], c[0]):
            continue

        distance = math.sqrt((c[1][0] - m[1][0]) ** 2 + (c[1][1] - m[1][1]) ** 2)
        if distance < min_parallax and m[2] <= c[2]:
            return False

    return True


# Get functions


def get_convexity_points(component):
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    hull = cv2.convexHull(contour, returnPoints=False)

    points = []

    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return points

    for i in range(defects.shape[0]):
        _, _, index, _ = defects[i, 0]
        point = tuple(contour[index][0])
        points.append(point)

    return points


def get_optimal_threshold(image, factor=1):
    flattened = image.flatten()
    mode = stats.mode(flattened)[0][0]
    median = int(np.median(flattened))
    threshold = int((mode + factor * median) / 2)

    return threshold


def fill_holes(mask):
    holes = np.where(mask == 0)

    if len(holes[0]) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    seed = (holes[0][0], holes[1][0])
    holes_mask_inverted = mask.copy()
    h_, w_ = mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
    cv2.floodFill(holes_mask_inverted, mask_, seedPoint=seed, newVal=255)
    holes_mask = cv2.bitwise_not(holes_mask_inverted)
    filled = mask + holes_mask

    return filled


# Get functions

def get_images_as_array(path):
    bw_file_names = glob.glob(path + "/*C0*")
    bw_file_names.sort()
    bw_images = [cv2.imread(img) for img in bw_file_names]

    color_file_names = glob.glob(path + "/*C1*")
    color_file_names.sort()
    color_images = [cv2.imread(img) for img in color_file_names]

    return bw_images, bw_file_names, color_images, color_file_names


def get_samples_as_array(path):
    samples_file_names = glob.glob(path + "/*")
    samples_file_names.sort()
    samples = [cv2.imread(img) for img in samples_file_names]

    return samples, samples_file_names


def get_component(labels, label):
    component = np.zeros_like(labels, dtype=np.uint8)
    component[labels == label] = 255

    return component


def get_biggest_component(image):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    max_area = float("-inf")
    biggest_component = None

    for i in range(1, retval):
        component = get_component(labels, i)
        component_area = cv2.countNonZero(component)

        if component_area > max_area:
            biggest_component = component
            max_area = component_area

    return biggest_component


def get_dominant_colors(image, clusters):
    L, a, b = cv2.split(image)
    ab = cv2.merge((a, b))

    # Reshaping to a list of pixels
    converted = ab.reshape((image.shape[0] * image.shape[1], 2))

    # Using k-means to cluster pixels
    kmeans = KMeans(n_clusters=clusters, n_init=100, max_iter=3000)
    kmeans.fit(converted)

    # The cluster centers are the dominant colors
    colors = kmeans.cluster_centers_.astype(int)

    # Save labels
    labels = kmeans.labels_.reshape(image.shape[0], image.shape[1])

    return colors, labels


def get_russet_index(colors):
    distances = []

    dark_brown_lab = rgb_to_lab(dark_brown_rgb)[1:3]

    for c in colors:
        d = dist.cityblock(c, dark_brown_lab)
        distances.append(d)

    min = [float("inf"), -1]

    for i in range(len(distances)):
        cur = distances[i]
        if cur <= min[0]:
            min[0] = cur
            min[1] = i

    russet_index = min[1]

    return russet_index


def get_clustering_sample(image, labels, index):
    mask = get_component(labels, index)
    mean = cv2.mean(image, mask)
    sample = [int(mean[0]), int(mean[1]), int(mean[2])]

    return sample


def get_mahalanobis_sample(samples, russet_sample):
    mean_tot = 0

    for s in samples:
        s_lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
        mean_tot += np.mean(s_lab, axis=(0, 1))[0]

    mean = mean_tot / len(samples)
    result = np.array([mean, russet_sample[0][0], russet_sample[0][1]])

    return result


# Drawing functions

def draw_fruit_outline(image, mask, thickness, color=(0, 255, 0)):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color, thickness)


def draw_defect(image, component, thickness, scale, min_area, max_area, min_parallax):
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return

    filtered_contours = filter_duplicated_contours(contours, min_parallax)

    drawn = 0

    for c in filtered_contours:
        area = cv2.contourArea(c)
        if min_area < area and len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            scaled_axes = (ellipse[1][0] * scale, ellipse[1][1] * scale)
            if scaled_axes[0] * scaled_axes[1] * math.pi < max_area:
                scaled_ellipse = ellipse[0], scaled_axes, ellipse[2]
                cv2.ellipse(image, scaled_ellipse, (0, 0, 255), thickness)
                drawn += 1

    return drawn


# Show functions

def show_image(name, image, x=0, y=0):
    cv2.imshow(name, image)
    cv2.moveWindow(name, x, y)
    cv2.waitKey()
    cv2.destroyAllWindows()


def show_sample_lab(lab, name, width, height):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = lab
    bgr = cv2.cvtColor(temp, cv2.COLOR_LAB2BGR)[0][0]

    sample = np.empty((width, height, 3), np.uint8)
    sample[:, :] = bgr

    cv2.imshow(name, sample)
    cv2.moveWindow(name, 255, 0)
