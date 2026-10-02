import cv2


def load_image(image_path):
    """
    Load image from the specified path.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image


def to_grayscale(image):
    """
    Convert BGR image to Grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray):
    """
    Apply Gaussian Blur to reduce noise.
    """
    return cv2.GaussianBlur(gray, (5, 5), 0)


def adaptive_threshold(blur):
    """
    Apply Adaptive Gaussian Thresholding.
    """
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )


def morphological_opening(threshold):
    """
    Remove small noise using Morphological Opening.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    opening = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    return opening