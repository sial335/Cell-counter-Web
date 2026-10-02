import cv2
import numpy as np

from preprocessing import (
    to_grayscale,
    gaussian_blur,
    adaptive_threshold,
    morphological_opening
)
from viability import classify_cell, background_blueness
from roi import clamp_square


# Colours (BGR) used on the annotated image
COLOUR_LIVE = (0, 200, 0)
COLOUR_DEAD = (0, 0, 255)
COLOUR_EXCLUDED = (0, 255, 255)


def remove_grid_lines(binary):
    """
    Delete long straight horizontal / vertical lines (hemocytometer
    grid lines) from a binary image. Without this, a grid line at the
    edge of the counting square forms a frame that hides every cell
    inside it.
    """
    H, W = binary.shape[:2]
    line_len = max(40, int(0.15 * min(H, W)))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def detect_cells_contour(
    image,
    square,
    mode="rbc",
    min_area=30,
    min_circularity=0.30,
    clump_ratio=1.5,
    max_clump_cells=6,
    min_solidity=0.80,
    viability_params=None
):
    """
    Count live and dead cells inside the counting square.

    image   : full BGR image
    square  : (x, y, w, h) counting square in pixels
    mode    : "rbc" or "pollen" (chooses default colour rules)

    Counting rules
    --------------
    * Only the pixels inside the square are analysed, so nothing
      outside the grid can be counted.
    * Standard hemocytometer border rule: cells touching the LEFT or
      TOP edge are counted, cells touching the RIGHT or BOTTOM edge
      are ignored.
    * Objects much larger than the median cell are treated as clumps
      and counted as round(area / median area) cells.

    Returns a dict with the annotated crop, counts and per-object data.
    """

    x0, y0, w0, h0 = clamp_square(square, image.shape)
    crop = image[y0:y0 + h0, x0:x0 + w0].copy()
    H, W = crop.shape[:2]

    viability_params = dict(viability_params or {})
    viability_params["blue_ref"] = background_blueness(crop)

    # ------------------------------------------------------
    # Preprocessing (same functions as before)
    # ------------------------------------------------------
    gray = to_grayscale(crop)
    blur = gaussian_blur(gray)
    threshold = adaptive_threshold(blur)
    opening = morphological_opening(threshold)
    opening = remove_grid_lines(opening)

    contours, _ = cv2.findContours(
        opening,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ------------------------------------------------------
    # Measure every contour
    # ------------------------------------------------------
    candidates = []
    max_area = 0.25 * W * H

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0

        bx, by, bw, bh = cv2.boundingRect(contour)
        touches_right_bottom = (bx + bw >= W - 1) or (by + bh >= H - 1)

        candidates.append({
            "contour": contour,
            "area": area,
            "circularity": circularity,
            "solidity": solidity,
            "excluded": touches_right_bottom
        })

    # Typical single-cell area = median of round, counted objects
    singles = [
        c["area"] for c in candidates
        if c["circularity"] >= min_circularity and not c["excluded"]
    ]
    median_area = float(np.median(singles)) if singles else 0.0

    # ------------------------------------------------------
    # Count
    # ------------------------------------------------------
    annotated = crop.copy()
    objects = []
    counts = {"live": 0, "dead": 0}
    excluded_border = 0
    clump_objects = 0

    for c in candidates:

        contour = c["contour"]

        # Right / bottom border rule
        if c["excluded"]:
            excluded_border += 1
            cv2.drawContours(annotated, [contour], -1, COLOUR_EXCLUDED, 1)
            continue

        n_cells = 1
        is_clump = False

        if median_area > 0 and c["area"] > clump_ratio * median_area:

            n_cells = int(round(c["area"] / median_area))

            # Very big / ragged objects are usually grid lines or debris
            if n_cells > max_clump_cells or c["solidity"] < min_solidity:
                continue

            is_clump = n_cells > 1

        elif c["circularity"] < min_circularity:
            continue

        label, hue, sat, blue = classify_cell(
            crop, contour, mode, viability_params
        )

        counts[label] += n_cells

        colour = COLOUR_DEAD if label == "dead" else COLOUR_LIVE
        cv2.drawContours(annotated, [contour], -1, colour, 2)

        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(contour)
            cx, cy = bx + bw // 2, by + bh // 2

        if is_clump:
            clump_objects += 1
            text = f"x{n_cells}"
            cv2.putText(annotated, text, (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(annotated, text, (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        objects.append({
            "id": len(objects) + 1,
            "x": cx + x0,
            "y": cy + y0,
            "area": round(c["area"], 1),
            "cells": n_cells,
            "hue": round(hue, 1),
            "saturation": round(sat, 1),
            "blueness": round(blue, 1),
            "label": label
        })

    total = counts["live"] + counts["dead"]

    return {
        "annotated": annotated,
        "total": total,
        "live": counts["live"],
        "dead": counts["dead"],
        "viability": (100.0 * counts["live"] / total) if total else 0.0,
        "objects": objects,
        "excluded_border": excluded_border,
        "clump_objects": clump_objects,
        "median_area": median_area,
        "square": (x0, y0, w0, h0)
    }



# Starting values for the circle detector (radius in pixels of the
# image after app.py shrinks it to 1600 px). Adjust with the sliders.
DEFAULT_DETECTION = {
    "rbc":    {"min_r": 8,  "max_r": 16, "sens": 18},
    "pollen": {"min_r": 24, "max_r": 44, "sens": 26},
}


def _circle_contour(cx, cy, r):
    pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(r), int(r)), 0, 0, 360, 10)
    return pts.reshape(-1, 1, 2).astype(np.int32)


def detect_cells_circles(
    image,
    square,
    mode="rbc",
    min_radius=8,
    max_radius=16,
    sensitivity=18,
    viability_params=None
):
    """
    Count cells as circles (Hough transform) inside the counting square.

    Works well for ring-shaped / pale cells and for touching cells,
    because every cell is found on its own. No clump correction needed.
    The one setting that matters is the radius range: it must bracket
    the radius of a SINGLE cell in pixels.
    """

    x0, y0, w0, h0 = clamp_square(square, image.shape)
    crop = image[y0:y0 + h0, x0:x0 + w0].copy()
    H, W = crop.shape[:2]

    viability_params = dict(viability_params or {})
    viability_params["blue_ref"] = background_blueness(crop)

    gray = cv2.GaussianBlur(to_grayscale(crop), (7, 7), 0)

    found = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, int(1.6 * min_radius)),
        param1=60,
        param2=sensitivity,
        minRadius=int(min_radius),
        maxRadius=int(max_radius)
    )

    circles = [] if found is None else [tuple(c) for c in found[0]]

    # Drop circles that mostly overlap a stronger one (Hough returns
    # the strongest first)
    kept = []
    for cx, cy, r in circles:
        if all(np.hypot(cx - kx, cy - ky) >= 0.7 * max(r, kr)
               for kx, ky, kr in kept):
            kept.append((cx, cy, r))

    annotated = crop.copy()
    objects = []
    counts = {"live": 0, "dead": 0}
    excluded_border = 0

    for cx, cy, r in kept:

        # Border rule: ignore cells touching the RIGHT or BOTTOM edge
        if cx + r > W - 1 or cy + r > H - 1:
            excluded_border += 1
            cv2.circle(annotated, (int(cx), int(cy)), int(r),
                       COLOUR_EXCLUDED, 1)
            continue

        # Colour is read from the inner 70 % of the circle
        label, hue, sat, blue = classify_cell(
            crop, _circle_contour(cx, cy, 0.7 * r), mode, viability_params
        )
        counts[label] += 1

        colour = COLOUR_DEAD if label == "dead" else COLOUR_LIVE
        cv2.circle(annotated, (int(cx), int(cy)), int(r), colour, 2)

        objects.append({
            "id": len(objects) + 1,
            "x": int(cx) + x0,
            "y": int(cy) + y0,
            "area": round(float(np.pi * r * r), 1),
            "cells": 1,
            "hue": round(hue, 1),
            "saturation": round(sat, 1),
            "blueness": round(blue, 1),
            "label": label
        })

    total = counts["live"] + counts["dead"]
    radii = [o for o in kept]
    median_r = float(np.median([c[2] for c in kept])) if kept else 0.0

    return {
        "annotated": annotated,
        "total": total,
        "live": counts["live"],
        "dead": counts["dead"],
        "viability": (100.0 * counts["live"] / total) if total else 0.0,
        "objects": objects,
        "excluded_border": excluded_border,
        "clump_objects": 0,
        "median_area": float(np.pi * median_r ** 2),
        "square": (x0, y0, w0, h0)
    }


def detect_cells(image, square, mode="rbc", method="circles", **kwargs):
    """
    method = "circles"  -> detect_cells_circles (recommended)
    method = "contours" -> detect_cells_contour (threshold + clump correction)
    """
    if method == "contours":
        allowed = ("min_area", "min_circularity", "clump_ratio",
                   "max_clump_cells", "min_solidity", "viability_params")
        return detect_cells_contour(
            image, square, mode,
            **{k: v for k, v in kwargs.items() if k in allowed}
        )

    allowed = ("min_radius", "max_radius", "sensitivity", "viability_params")
    return detect_cells_circles(
        image, square, mode,
        **{k: v for k, v in kwargs.items() if k in allowed}
    )


# ==========================================================
# LEGACY FUNCTION (only needed by the old main.py)
# ==========================================================

def detect_rbc(roi_binary, roi_original):
    """
    Old contour-based counter. app.py uses detect_cells() instead.
    """

    contours, _ = cv2.findContours(
        roi_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = roi_original.copy()
    rbc_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 30 or area > 1500:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.30:
            continue

        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        rbc_count += 1

    return output, rbc_count