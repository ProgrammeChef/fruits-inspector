import cv2
import utils


def run():
    # Read and store all images into an array
    path = "./images/first_task"
    bw_images, bw_file_names, color_images, color_file_names = utils.get_images_as_array(path)

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

        # Binarize the image to separate foreground and background
        threshold, binarized = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Get fruit mask (biggest component)
        mask = utils.get_biggest_component(binarized)

        # Fill the holes
        filled = utils.fill_holes(mask)

        # Get colored fruit from mask
        fruit = cv2.bitwise_and(bw_images[i], bw_images[i], mask=filled)

        # Apply a bilateral blur to remove noise but preserving edges
        blur = cv2.bilateralFilter(fruit, 5, 100, 75)

        # Perform a Canny edge detection
        canny = utils.auto_canny(blur, 0.7)

        # Apply a closing operation to consolidate detected edges
        structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, structuringElement)

        # Perform a connected components labeling to detect components
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 4)

        # Get corresponding colored image
        display = color_images[i]

        # Outline the fruit using the binary mask
        utils.outline_fruit(display, filled, 1)

        # Iterate over the detected components to isolate defects
        # (range starts from 2 to exclude outer background and the fruit itself)
        for j in range(2, retval):
            # Isolate current binarized component
            component = utils.get_component(labels, j)
            utils.draw_defect(display, component, 2, 2.2)

        # Show processed image
        utils.show_image(bw_file_name, closed, 0, 0)

        # Print detected defects number
        print(color_file_name + ": detected " + str(retval - 2) + " defect(s)")

        # Show original image highlighting defects
        utils.show_image(color_file_name, display, 0, 0)
