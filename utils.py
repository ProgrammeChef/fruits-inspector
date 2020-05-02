import glob
import cv2
import numpy as np


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

def show_image(name, image, x, y):
    cv2.imshow(name, image)
    cv2.moveWindow(name, x, y)
    cv2.waitKey()
    cv2.destroyAllWindows()


def draw_defect(image, component, thickness, scale):
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    scaled_axes = (ellipse[1][0] * scale, ellipse[1][1] * scale)
    scaled_ellipse = ellipse[0], scaled_axes, ellipse[2]
    cv2.ellipse(image, scaled_ellipse, (0, 0, 255), thickness)


def outline_fruit(image, mask, thickness):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), thickness)
