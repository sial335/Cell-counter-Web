import cv2
import numpy as np


# ==========================================================
# LIVE / DEAD RULES
#
# POLLEN (trypan blue: dead grains take up the blue dye)
#
#   rule = "blueness"  (recommended)
#       blueness = 100 * ln(Blue / Red) inside the grain
#       (roughly "% bluer than red"). Every image has its own colour
#       cast / white balance, so the blueness of the slide BACKGROUND
#       is measured and subtracted first. A grain is DEAD if it is
#       bluer than the background by at least "blue_margin" points.
#       Brown live grains are always much less blue than the
#       background, so no per-image tuning is needed. Using a ratio
#       (log) cancels camera gain / white-balance differences.
#
#   rule = "hsv"
#       DEAD if mean hue is inside [dead_hue_min, dead_hue_max]
#       AND saturation >= sat_min (older rule, needs more tuning).
#
# RBC were NOT stained, so live/dead is not assessed by colour.
# (app.py reports the total count only for RBC.)
# ==========================================================

DEFAULT_PARAMS = {
    "pollen": {
        "rule": "blueness",
        "blue_margin": 10,
        "dead_hue_min": 85,
        "dead_hue_max": 135,
        "sat_min": 60,
    },
    "rbc": {
        "rule": "hsv",
        "blue_margin": 10,
        "dead_hue_min": 95,
        "dead_hue_max": 135,
        "sat_min": 60,
    },
}


def _cell_pixels(image_bgr, contour):
    """
    Return the BGR and HSV pixels (N x 3, float) of the inner part of
    a cell, or (None, None) if the cell is empty.
    """
    x, y, w, h = cv2.boundingRect(contour)

    patch = image_bgr[y:y + h, x:x + w]
    if patch.size == 0:
        return None, None

    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    shifted = contour - np.array([x, y], dtype=contour.dtype)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    # Ignore the cell edge, which is blurred with the background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(mask, kernel)
    if cv2.countNonZero(inner) > 0:
        mask = inner

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    bgr_pixels = patch[mask > 0].astype(np.float64)
    hsv_pixels = hsv[mask > 0].astype(np.float64)

    if len(bgr_pixels) == 0:
        return None, None

    return bgr_pixels, hsv_pixels


def cell_colour(image_bgr, contour):
    """
    Return (hue, saturation, value) of the pixels inside a contour.

    Hue is averaged on a circle (weighted by saturation) so that
    reds near 0 and 179 do not average out to blue.
    """
    _, pixels = _cell_pixels(image_bgr, contour)

    if pixels is None:
        return 0.0, 0.0, 0.0

    angles = np.deg2rad(pixels[:, 0] * 2.0)
    weights = pixels[:, 1] + 1e-6

    mean_sin = (weights * np.sin(angles)).sum() / weights.sum()
    mean_cos = (weights * np.cos(angles)).sum() / weights.sum()

    hue = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0) / 2.0
    sat = pixels[:, 1].mean()
    val = pixels[:, 2].mean()

    return float(hue), float(sat), float(val)


def cell_blueness(image_bgr, contour):
    """
    100 * ln(Blue / Red) inside a cell.
    Positive = bluish, negative = brownish.
    """
    pixels, _ = _cell_pixels(image_bgr, contour)

    if pixels is None:
        return 0.0

    blue = pixels[:, 0].mean() + 1.0
    red = pixels[:, 2].mean() + 1.0

    return float(100.0 * np.log(blue / red))


def background_blueness(image_bgr):
    """
    Blueness (100 * ln(Blue / Red)) of the slide background, measured
    on the brighter half of the image where there are no cells.
    """
    img = image_bgr.astype(np.float32)

    log_ratio = 100.0 * np.log((img[:, :, 0] + 1.0) / (img[:, :, 2] + 1.0))
    brightness = img.mean(axis=2)

    bright = brightness >= np.median(brightness)

    return float(np.median(log_ratio[bright]))


def classify_cell(image_bgr, contour, mode="rbc", params=None):
    """
    Classify one cell as "live" or "dead".

    Returns (label, hue, saturation, blueness_vs_background).
    """
    rules = dict(DEFAULT_PARAMS.get(mode, DEFAULT_PARAMS["rbc"]))
    if params:
        rules.update(params)

    hue, sat, _ = cell_colour(image_bgr, contour)

    blue = cell_blueness(image_bgr, contour) - rules.get("blue_ref", 0.0)

    if rules.get("rule") == "blueness":
        is_dead = blue >= rules["blue_margin"]
    else:
        is_dead = (
            rules["dead_hue_min"] <= hue <= rules["dead_hue_max"]
            and sat >= rules["sat_min"]
        )

    return ("dead" if is_dead else "live"), hue, sat, blue