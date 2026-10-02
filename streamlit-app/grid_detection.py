import cv2
import numpy as np


def detect_grid(roi_original):
    """
    Detect horizontal and vertical haemocytometer grid lines.

    Returns:
        horizontal: detected horizontal lines
        vertical: detected vertical lines
        grid: combined grid
    """

    # Convert ROI to grayscale
    gray = cv2.cvtColor(
        roi_original,
        cv2.COLOR_BGR2GRAY
    )

    # Threshold bright grid lines
    _, thresh = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # ======================================================
    # HORIZONTAL LINES
    # ======================================================

    horizontal = thresh.copy()

    horizontal_size = max(
        10,
        horizontal.shape[1] // 20
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_size, 1)
    )

    horizontal = cv2.erode(
        horizontal,
        horizontal_kernel
    )

    horizontal = cv2.dilate(
        horizontal,
        horizontal_kernel
    )

    # ======================================================
    # VERTICAL LINES
    # ======================================================

    vertical = thresh.copy()

    vertical_size = max(
        10,
        vertical.shape[0] // 20
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_size)
    )

    vertical = cv2.erode(
        vertical,
        vertical_kernel
    )

    vertical = cv2.dilate(
        vertical,
        vertical_kernel
    )

    # ======================================================
    # COMBINE GRID
    # ======================================================

    grid = cv2.add(
        horizontal,
        vertical
    )

    return horizontal, vertical, grid


def get_grid_bounds(horizontal, vertical, roi_original, margin=5):
    """
    Estimate the rectangular haemocytometer grid boundary.

    The grid is NOT forced to be square.

    Returns:
        x, y, w, h
    """

    H, W = roi_original.shape[:2]

    # ------------------------------------------------------
    # Find horizontal-line pixels
    # ------------------------------------------------------

    horizontal_projection = np.sum(
        horizontal > 0,
        axis=1
    )

    # ------------------------------------------------------
    # Find vertical-line pixels
    # ------------------------------------------------------

    vertical_projection = np.sum(
        vertical > 0,
        axis=0
    )

    # Minimum number of pixels required to consider
    # a row/column as part of a grid line.
    horizontal_threshold = max(
        20,
        int(W * 0.10)
    )

    vertical_threshold = max(
        20,
        int(H * 0.10)
    )

    horizontal_indices = np.where(
        horizontal_projection >= horizontal_threshold
    )[0]

    vertical_indices = np.where(
        vertical_projection >= vertical_threshold
    )[0]

    # ------------------------------------------------------
    # Safety fallback
    # ------------------------------------------------------

    if (
        len(horizontal_indices) < 2
        or len(vertical_indices) < 2
    ):
        print(
            "Warning: grid boundary could not be detected."
        )

        return (
            0,
            0,
            W,
            H
        )

    # ------------------------------------------------------
    # Grid boundaries
    # ------------------------------------------------------

    y_min = int(horizontal_indices.min())
    y_max = int(horizontal_indices.max())

    x_min = int(vertical_indices.min())
    x_max = int(vertical_indices.max())

    # Add a small margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)

    x_max = min(W - 1, x_max + margin)
    y_max = min(H - 1, y_max + margin)

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    return (
        x_min,
        y_min,
        w,
        h
    )


def crop_grid(roi_original, bounds):
    """
    Crop the detected haemocytometer grid.

    The rectangular dimensions are preserved.
    """

    x, y, w, h = bounds

    H, W = roi_original.shape[:2]

    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))

    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))

    return roi_original[
        y:y + h,
        x:x + w
    ].copy()


def draw_grid_overlay(roi_original, bounds):
    """
    Draw the detected grid boundary on the original ROI.

    Useful for checking whether automatic adjustment
    is correct.
    """

    overlay = roi_original.copy()

    x, y, w, h = bounds

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    return overlay