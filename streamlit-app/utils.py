import cv2


def show_image(window_name, image, width=800, height=600):
    """
    Display an image in a resizable window.
    """

    display = cv2.resize(image, (width, height))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.imshow(window_name, display)


def save_image(file_path, image):
    """
    Save an image to the output folder.
    """

    cv2.imwrite(file_path, image)