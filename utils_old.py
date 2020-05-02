import glob
import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment

# This constant is based on the maximum empirically detected area: 5293
MAX_ROD_AREA = 6000

# An array of random colors to visualise correctness of components labeling
random_colors = [(255, 0, 0), (0, 0, 255), (255, 127, 255), (127, 0, 255), (127, 0, 127), (0, 255, 0), (255, 255, 0),
                 (0, 255, 255), (255, 0, 255), ]


# Processing functions

def median_blur(image, kernel_size, iterations):
    for i in range(iterations):
        image = cv2.medianBlur(image, kernel_size)

    return image


def color_component(image, labels, label):
    image[labels == label] = [random_colors[label - 1][0], random_colors[label - 1][1], random_colors[label - 1][2]]

    return image


def separate_component(component, binarized):
    points = get_convexity_points(component)

    filtered_points = filter_points(binarized, points, 9, 1.5)

    sorted_points = sort_points_pairwise(filtered_points)

    for p in sorted_points:
        # We impose a maximum distance threshold to avoid certainly wrong connections (customisable)
        if p[0] <= 70:
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


def separate_touching_objects(binarized, timeout):
    while True:
        separated = 0
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, 4)

        for k in range(1, retval):
            # Isolate current binarized component
            component = get_component(labels, k)

            if cv2.countNonZero(component) > MAX_ROD_AREA:
                binarized = separate_component(component, binarized)
                separated += 1

        if separated == 0 or timeout == 0:
            break

        timeout -= 1

    return binarized


# Boolean functions

def is_not_a_rod(length, width):
    if width == 0:
        return True

    elongation = length / width

    if 1 < elongation < 2 or elongation > 5:
        return True


# Get functions

def get_images_as_array():
    file_names = glob.glob("images/*.bmp")
    file_names.sort()
    images = [cv2.imread(img, 0) for img in file_names]

    return images, file_names


def get_component(labels, label):
    component = np.zeros_like(labels, dtype=np.uint8)
    component[labels == label] = 255

    return component


def get_angle(moments):
    if (moments['mu20'] - moments['mu02']) == 0:
        return None

    theta = -0.5 * math.atan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))
    d2theta = 2 * (moments['mu20'] - moments['mu02']) * math.cos(2 * theta) - 4 * moments['mu11'] * math.sin(
        2 * theta)

    if d2theta > 0:
        return theta
    else:
        return theta + math.pi / 2


def get_oriented_mer(component, angle, centroid):
    alpha = -math.sin(angle)
    beta = math.cos(angle)
    major = (alpha, -beta, beta * centroid[1] - alpha * centroid[0])
    minor = (beta, alpha, -beta * centroid[0] - alpha * centroid[1])

    # MER contact points to be detected
    c1 = c2 = c3 = c4 = (0, 0)

    max_c1_maj = max_c3_min = float("-inf")
    min_c2_maj = min_c4_min = float("inf")

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for p in contours[0]:
        x = p[0][0]
        y = p[0][1]

        dist_maj = (major[0] * x + major[1] * y + major[2]) / math.sqrt(major[0] ** 2 + major[1] ** 2)
        dist_min = (minor[0] * x + minor[1] * y + minor[2]) / math.sqrt(minor[0] ** 2 + minor[1] ** 2)

        # If distance from the major axis is greater than the maximum ever encountered
        if dist_maj > max_c1_maj:
            c1 = (x, y)
            max_c1_maj = dist_maj
        # If distance from the major axis is less than the minimum ever encountered
        if dist_maj < min_c2_maj:
            c2 = (x, y)
            min_c2_maj = dist_maj
        # If distance from the minor axis is greater than the maximum ever encountered
        if dist_min > max_c3_min:
            c3 = (x, y)
            max_c3_min = dist_min
        # If distance from the minor axis is less than the minimum ever encountered
        if dist_min < min_c4_min:
            c4 = (x, y)
            min_c4_min = dist_min

    line_c1 = (alpha / beta, c1[1] - alpha / beta * c1[0])
    line_c2 = (alpha / beta, c2[1] - alpha / beta * c2[0])
    line_c3 = (-beta / alpha, c3[1] + beta / alpha * c3[0])
    line_c4 = (-beta / alpha, c4[1] + beta / alpha * c4[0])

    temp_x = (line_c1[1] - line_c3[1]) / (line_c3[0] - line_c1[0])
    v1 = (temp_x, line_c1[0] * temp_x + line_c1[1])
    temp_x = (line_c1[1] - line_c4[1]) / (line_c4[0] - line_c1[0])
    v2 = (temp_x, line_c1[0] * temp_x + line_c1[1])

    temp_x = (line_c2[1] - line_c3[1]) / (line_c3[0] - line_c2[0])
    v3 = (temp_x, line_c2[0] * temp_x + line_c2[1])
    temp_x = (line_c2[1] - line_c4[1]) / (line_c4[0] - line_c2[0])
    v4 = (temp_x, line_c2[0] * temp_x + line_c2[1])

    mer = (v1, v2, v3, v4)

    length = math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
    width = math.sqrt((v1[0] - v3[0]) ** 2 + (v1[1] - v3[1]) ** 2)

    return mer, length, width


def get_barycenter_width(component, angle, centroid):
    alpha = -math.sin(angle)
    beta = math.cos(angle)
    major = (alpha, -beta, beta * centroid[1] - alpha * centroid[0])
    minor = (beta, alpha, -beta * centroid[0] - alpha * centroid[1])

    bp1 = bp2 = (0, 0)
    min_bp1 = min_bp2 = float("inf")

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for p in contours[0]:
        x = p[0][0]
        y = p[0][1]
        dist_maj = (major[0] * x + major[1] * y + major[2]) / math.sqrt(major[0] ** 2 + major[1] ** 2)
        dist_min = (minor[0] * x + minor[1] * y + minor[2]) / math.sqrt(minor[0] ** 2 + minor[1] ** 2)

        # If the absolute distance from the minor axis is less than the minimum ever encountered
        # for bp1 AND the distance from the major axis is positive
        if abs(dist_min) < min_bp1 and dist_maj > 0:
            bp1 = (x, y)
            min_bp1 = abs(dist_min)
        # If the absolute distance from the minor axis is less than the minimum ever encountered
        # for bp2 AND the distance from the major axis is negative
        if abs(dist_min) < min_bp2 and dist_maj < 0:
            bp2 = (x, y)
            min_bp2 = abs(dist_min)

    width = math.sqrt((bp1[0] - bp2[0]) ** 2 + (bp1[1] - bp2[1]) ** 2)
    bar_points = [bp1, bp2]

    return bar_points, width


def get_convexity_points(component):
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    points = []

    for i in range(defects.shape[0]):
        _, _, index, _ = defects[i, 0]
        point = tuple(contour[index][0])
        points.append(point)

    return points


def get_hole_diameter(area):
    diameter = 2 * math.sqrt(area / math.pi)
    return diameter


# Drawing functions

def draw_centroid(image, centroid):
    cv2.circle(image, (int(centroid[0]), int(centroid[1])), 2, (255, 255, 255), -1)


def draw_barycenter_width(image, points):
    p1, p2 = points[0], points[1]

    cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), 1, cv2.LINE_AA)


def draw_oriented_mer(image, mer):
    v1, v2, v3, v4 = mer[0], mer[1], mer[2], mer[3]

    cv2.line(image, (int(v1[0]), int(v1[1])), (int(v3[0]), int(v3[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v4[0]), int(v4[1])), (int(v2[0]), int(v2[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v2[0]), int(v2[1])), (int(v1[0]), int(v1[1])), (255, 255, 255), 1, cv2.LINE_AA)


def draw_orientation_axis(image, centroid, angle, length):
    length = int(length * 1.3 / 2)
    alpha = -math.sin(angle)
    beta = math.cos(angle)

    p1 = (int(centroid[0] + length * beta),
          int(centroid[1] + length * alpha))

    p2 = (int(centroid[0] - length * beta),
          int(centroid[1] - length * alpha))

    cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), 1, cv2.LINE_AA)


# Print functions

def print_rod_info(centroid, mer, angle, length, width, bar_width, holes):
    string = ""
    rod_type = "Not a rod"

    if len(holes) == 1:
        rod_type = "A"
    elif len(holes) == 2:
        rod_type = "B"

    string += "Rod type:\t" + rod_type + "\n"
    string += "Position:\tCentroid = (" + "{:.2f}".format(centroid[0]) + ", " + "{:.2f}".format(
        centroid[1]) + ")\n\t\tMER = ((" + "{:.2f}".format(mer[0][0]) + ", " + "{:.2f}".format(
        mer[0][1]) + "), (" + "{:.2f}".format(mer[1][0]) + ", " + "{:.2f}".format(
        mer[1][1]) + "), (" + "{:.2f}".format(mer[2][0]) + ", " + "{:.2f}".format(mer[2][1]) + "), (" + "{:.2f}".format(
        mer[3][0]) + ", " + "{:.2f}".format(mer[3][1]) + "))\n"
    string += "Orientation:\t" + "{:.2f}".format(angle * 180 / math.pi) + " deg\n"
    string += "Size:\t\tLength = " + "{:.2f}".format(length) + "\n\t\tWidth = " + "{:.2f}".format(
        width) + "\n\t\tBarycenter width: " + "{:.2f}".format(
        bar_width) + "\n"
    string += "Holes:\t\t"

    if len(holes) == 0:
        string += "None"
    else:
        for i, h in enumerate(holes):
            if i > 0:
                string += "\t\t"

            string += "Hole " + str(i + 1) + ":\tCentroid = (" + \
                      "{:.2f}".format(h[0][0]) + ", " + \
                      "{:.2f}".format(h[0][1]) + ")\n\t\t\tDiameter: " + \
                      "{:.2f}".format(h[1]) + "\n"

    print(string)


def print_image_info(file_name):
    string = "==============================\n"
    string += file_name + "\n"
    string += "==============================\n"

    print(string)


# Show functions

def show_image(name, image, x, y):
    cv2.imshow(name, image)
    cv2.moveWindow(name, x, y)
    cv2.waitKey()
    cv2.destroyAllWindows()
