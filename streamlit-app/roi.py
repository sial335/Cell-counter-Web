import cv2
import numpy as np


# ==========================================================
# COUNTING SQUARE HELPERS  (used by app.py)
# ==========================================================

def clamp_square(square, shape, min_size=20):
    """
    Keep the counting square (x, y, w, h) inside the image.
    """
    H, W = shape[:2]
    x, y, w, h = [int(v) for v in square]

    x = max(0, min(x, W - min_size))
    y = max(0, min(y, H - min_size))
    w = max(min_size, min(w, W - x))
    h = max(min_size, min(h, H - y))

    return x, y, w, h


def square_from_percent(shape, left, top, width, height):
    """
    Convert slider values (0-100 % of the image) to pixel coordinates.
    """
    H, W = shape[:2]

    x = int(W * left / 100)
    y = int(H * top / 100)
    w = int(W * width / 100)
    h = int(H * height / 100)

    return clamp_square((x, y, w, h), shape)


def suggest_square(shape, fraction=0.5):
    """
    Suggest a centred square covering `fraction` of the shorter side.
    """
    H, W = shape[:2]
    side = int(min(H, W) * fraction)

    x = (W - side) // 2
    y = (H - side) // 2

    return clamp_square((x, y, side, side), shape)


def crop_to_square(image, square):
    """
    Return the part of the image inside the counting square.
    """
    x, y, w, h = clamp_square(square, image.shape)
    return image[y:y + h, x:x + w].copy()


# ==========================================================
# MICROSCOPE CIRCLE ROI  (kept so the old main.py still runs)
# ==========================================================

def extract_roi(image, gray, opening):
    """
    Detect the microscope field and extract the circular ROI.
    Falls back to the full image if no circle is found.
    """

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=1000,
        param1=100,
        param2=30,
        minRadius=300,
        maxRadius=700
    )

    if circles is None:
        print("Warning: microscope circle not detected, using full image.")
        return image.copy(), opening.copy(), image.copy()

    x, y, r = np.round(circles[0][0]).astype(int)

    circle_image = image.copy()
    cv2.circle(circle_image, (x, y), r, (0, 255, 0), 3)
    cv2.circle(circle_image, (x, y), 3, (0, 0, 255), -1)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    roi_binary = cv2.bitwise_and(opening, opening, mask=mask)
    roi_original = cv2.bitwise_and(image, image, mask=mask)

    return circle_image, roi_binary, roi_original
