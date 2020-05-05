import glob
import cv2
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial import distance
import math

dark_brown_rgb = [71, 56, 27]


def get_optimal_threshold(image):
    flattened = image.flatten()
    mode = stats.mode(flattened)[0][0]
    median = int(np.median(flattened))
    threshold = int((mode + median) / 2)

    return threshold


def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # Return the edged image
    return edged


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


# Show functions

def show_image(name, image, x=0, y=0):
    cv2.imshow(name, image)
    cv2.moveWindow(name, x, y)
    cv2.waitKey()
    cv2.destroyAllWindows()


def draw_defect(image, component, thickness, scale, min_area, max_area, min_parallax):
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

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
                drawn += 1
                scaled_ellipse = ellipse[0], scaled_axes, ellipse[2]
                cv2.ellipse(image, scaled_ellipse, (0, 0, 255), thickness)

    return drawn


def is_not_duplicate(m, measured, min_parallax):
    for c in measured:
        if np.array_equal(m[0], c[0]):
            continue

        dist = math.sqrt((c[1][0] - m[1][0]) ** 2 + (c[1][1] - m[1][1]) ** 2)
        if dist < min_parallax and m[2] < c[2]:
            return False

    return True


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


def outline_fruit(image, mask, thickness):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), thickness)


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

    average_labs = []

    for i in range(clusters):
        mask = get_component(labels, i)
        mean = cv2.mean(image, mask)
        avg = [int(mean[0]), int(mean[1]), int(mean[2])]
        average_labs.append(avg)

    return average_labs, colors, labels


def get_russet_index(colors):
    distances = []

    dark_brown_lab = rgb_to_lab(dark_brown_rgb)

    for c in colors:
        d = distance.cityblock(c, dark_brown_lab)
        distances.append(d)

    min = [float("inf"), -1]

    for i in range(len(distances)):
        cur = distances[i]
        if cur <= min[0]:
            min[0] = cur
            min[1] = i

    russet_index = min[1]

    return russet_index


def rgb_to_lab(rgb):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = rgb
    result = cv2.cvtColor(temp, cv2.COLOR_RGB2LAB)[0][0]

    return result


def bgr_to_lab(bgr):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = bgr
    result = cv2.cvtColor(temp, cv2.COLOR_BGR2LAB)[0][0]

    return result


def lab_to_rgb(lab):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = lab
    result = cv2.cvtColor(temp, cv2.COLOR_LAB2RGB)[0][0]

    return result


def show_sample_lab(lab, name, width, height):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = lab
    bgr = cv2.cvtColor(temp, cv2.COLOR_LAB2BGR)[0][0]

    sample = np.empty((width, height, 3), np.uint8)
    sample[:, :] = bgr

    cv2.imshow(name, sample)
    cv2.moveWindow(name, 255, 0)


def show_sample_rgb(rgb, name, width, height):
    temp = np.empty((1, 1, 3), np.uint8)
    temp[0] = rgb
    bgr = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)[0][0]

    sample = np.empty((width, height, 3), np.uint8)
    sample[:, :] = bgr

    show_image(name, sample)


def show_sample_bgr(bgr, name, width, height):
    sample = np.empty((width, height, 3), np.uint8)
    sample[:, :] = bgr

    show_image(name, sample)


def median_blur(image, kernel_size, iterations):
    for i in range(iterations):
        image = cv2.medianBlur(image, kernel_size)

    return image
