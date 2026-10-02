import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from roi import square_from_percent
from cell_detection import detect_cells, DEFAULT_DETECTION
from viability import DEFAULT_PARAMS

# Optional: automatic grid preview (uses YOUR grid_detection.py if present)
try:
    from grid_detection import (
        detect_grid,
        get_grid_bounds,
        crop_grid,
        draw_grid_overlay
    )
    HAS_GRID = True
except ImportError:
    HAS_GRID = False


MAX_SIDE = 1600   # large phone photos are shrunk to this size


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(page_title="Cell Counter", layout="wide")

st.title("RBC & Pollen Cell Counter")
st.caption(
    "Runs fully offline. Capture or upload a hemocytometer image, "
    "position the counting square, and read live / dead / total counts."
)


# ==========================================================
# HELPERS
# ==========================================================

def load_image(file):
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        return None

    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


LOG_FILE = Path(__file__).parent / "validation_log.csv"

LOG_COLUMNS = [
    "timestamp", "image", "cell_type",
    "app_total", "app_live", "app_dead",
    "manual_total", "manual_live", "manual_dead",
    "method", "radius_min", "radius_max", "sensitivity",
    "hue_lo", "hue_hi", "sat_min",
    "colour_rule", "blue_margin",
    "sq_left", "sq_top", "sq_width", "sq_height"
]


def save_to_log(row):
    """
    Append one row to validation_log.csv (created if missing).

    The log is NEVER renamed or split into several files. If the file
    was made by an older version (different columns) or was re-saved
    by Excel, it is upgraded in place: every old value is kept and the
    new columns are added empty.
    """
    old_rows = []
    header = None

    if LOG_FILE.exists():
        # utf-8-sig also handles the invisible mark Excel adds
        with open(LOG_FILE, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = [h.strip() for h in (reader.fieldnames or [])]
            for r in reader:
                old_rows.append({
                    (k or "").strip(): v for k, v in r.items()
                })

    if header == LOG_COLUMNS:
        # Normal case: just add one line
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)
        return

    # New file, or old/edited file: rewrite once with the current columns
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in LOG_COLUMNS})
        writer.writerow(row)


# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================

st.sidebar.header("Sample")

mode = st.sidebar.radio(
    "Cell type",
    ["rbc", "pollen"],
    format_func=lambda m: "Red blood cells (RBC)" if m == "rbc" else "Pollen"
)

defaults = DEFAULT_PARAMS[mode]

colour_rule = "hsv"
blue_margin = 10
hue_lo, hue_hi, sat_min = defaults["dead_hue_min"], defaults["dead_hue_max"], defaults["sat_min"]

if mode == "pollen":
    st.sidebar.header("Live / dead colour rule (trypan blue)")

    colour_rule = st.sidebar.radio(
        "Colour rule",
        ["blueness", "hsv"],
        format_func=lambda r: "Relative blueness (recommended)"
        if r == "blueness" else "Hue + saturation (manual)"
    )

    if colour_rule == "blueness":
        st.sidebar.caption(
            "A grain is DEAD if it is bluer than the slide background by "
            "at least this margin. Works across lighting / camera "
            "differences, so you normally do NOT need to change it. "
            "Raise it if live grains turn red; lower it if pale-blue "
            "dead grains stay green."
        )
        blue_margin = st.sidebar.slider(
            "Blueness margin (higher = fewer dead)",
            0, 60, 10,
            key="blue_margin"
        )
    else:
        st.sidebar.caption(
            "A grain is DEAD if its hue is inside this range and it is "
            "coloured enough."
        )
        hue_lo, hue_hi = st.sidebar.slider(
            "Dead-cell hue range (OpenCV 0-179)",
            0, 179,
            (defaults["dead_hue_min"], defaults["dead_hue_max"]),
            key=f"hue_{mode}"
        )
        sat_min = st.sidebar.slider(
            "Minimum colour saturation",
            0, 255, defaults["sat_min"],
            key=f"sat_{mode}"
        )
else:
    st.sidebar.info(
        "RBCs were not stained, so only the total count is reported."
    )
    # A saturation limit above the maximum means no cell is ever "dead"
    hue_lo, hue_hi, sat_min = 0, 179, 999

st.sidebar.header("Detection")

method = st.sidebar.radio(
    "Method",
    ["circles", "contours"],
    format_func=lambda m: "Circle detection (recommended)"
    if m == "circles" else "Contour + clump correction"
)

det = DEFAULT_DETECTION[mode]

# defaults so every name exists for both methods
r_lo, r_hi, sensitivity = det["min_r"], det["max_r"], det["sens"]
min_area, min_circ, clump_ratio = 30, 0.30, 1.5

if method == "circles":
    st.sidebar.caption(
        "Set the radius range so it brackets ONE cell (in pixels). "
        "Circles on the image should fit the cells."
    )
    r_lo, r_hi = st.sidebar.slider(
        "Cell radius range (px)", 4, 120,
        (det["min_r"], det["max_r"]),
        key=f"rad_{mode}"
    )
    sensitivity = st.sidebar.slider(
        "Sensitivity (lower = finds more cells)", 8, 60, det["sens"],
        key=f"sens_{mode}"
    )
else:
    min_area = st.sidebar.slider("Minimum cell area (px)", 10, 500, 30)
    min_circ = st.sidebar.slider("Minimum circularity", 0.10, 0.90, 0.30, 0.05)
    clump_ratio = st.sidebar.slider(
        "Clump threshold (x median cell)", 1.2, 3.0, 1.5, 0.1
    )

st.sidebar.header("Concentration")

dilution = st.sidebar.number_input(
    "Dilution factor (total volume / sample volume)",
    min_value=1.0, value=1.0, step=1.0
)
squares = st.sidebar.number_input(
    "Number of squares counted",
    min_value=1, value=1, step=1
)


# ==========================================================
# IMAGE INPUT
# ==========================================================

source = st.radio("Image source", ["Camera", "Upload"], horizontal=True)

if source == "Camera":
    file = st.camera_input("Capture the slide")
else:
    file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

if file is None:
    st.info("Capture or upload an image to begin.")
    st.stop()

image = load_image(file)

if image is None:
    st.error("This file could not be read as an image.")
    st.stop()


# ==========================================================
# AUTOMATIC GRID PREVIEW (optional, does not affect the count)
# ==========================================================

if HAS_GRID and st.checkbox("Show automatic grid detection preview"):
    try:
        horizontal, vertical, grid = detect_grid(image)
        grid_bounds = get_grid_bounds(horizontal, vertical, image)

        st.image(
            draw_grid_overlay(image, grid_bounds),
            channels="BGR",
            caption="Automatically detected grid boundary"
        )
        st.image(
            crop_grid(image, grid_bounds),
            channels="BGR",
            caption="Automatically cropped grid"
        )
    except Exception as e:
        st.warning(f"Grid preview failed: {e}")


# ==========================================================
# COUNTING SQUARE
# ==========================================================

st.subheader("1. Choose the counting square")
st.caption(
    "Move the green box so it exactly covers ONE counting square "
    "(the one bounded by the triple lines). Only cells inside it are counted."
)

c1, c2, c3, c4 = st.columns(4)
left = c1.slider("Left (%)", 0, 95, 25)
top = c2.slider("Top (%)", 0, 95, 25)
width = c3.slider("Width (%)", 5, 100, 50)
height = c4.slider("Height (%)", 5, 100, 50)

square = square_from_percent(image.shape, left, top, width, height)
x, y, w, h = square

preview = image.copy()
cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

st.image(preview, channels="BGR", caption="Counting square (green)")


# ==========================================================
# DETECTION
# ==========================================================

result = detect_cells(
    image,
    square,
    mode=mode,
    method=method,
    min_radius=r_lo,
    max_radius=r_hi,
    sensitivity=sensitivity,
    min_area=min_area,
    min_circularity=min_circ,
    clump_ratio=clump_ratio,
    viability_params={
        "rule": colour_rule,
        "blue_margin": blue_margin,
        "dead_hue_min": hue_lo,
        "dead_hue_max": hue_hi,
        "sat_min": sat_min
    }
)

st.subheader("2. Results")

if mode == "pollen":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cells", result["total"])
    m2.metric("Live cells", result["live"])
    m3.metric("Dead cells", result["dead"])
    m4.metric("Viability", f"{result['viability']:.1f} %")
else:
    st.metric("Total RBC", result["total"])
    st.caption("RBCs are unstained, so live/dead is not assessed.")

# Standard hemocytometer formula: 1 mm x 1 mm x 0.1 mm = 0.1 microlitre
concentration = (result["total"] / squares) * dilution * 1e4
st.write(f"**Concentration:** {concentration:,.0f} cells/mL "
         f"(count / squares x dilution x 10^4)")

if method == "contours":
    st.caption(
        f"Clumps corrected: {result['clump_objects']}  |  "
        f"Ignored (touching right/bottom border): {result['excluded_border']}"
    )
else:
    st.caption(
        f"Ignored (touching right/bottom border): {result['excluded_border']}"
    )

st.image(
    result["annotated"],
    channels="BGR",
    caption="Green = counted (red = dead pollen), yellow = ignored by the border rule, "
            "xN = clump counted as N cells"
)


# ==========================================================
# OPTIONAL VALIDATION AGAINST MANUAL COUNT
# ==========================================================

manual = st.number_input(
    "Manual total count for this square (optional, for validation)",
    min_value=0, value=0, step=1
)

if manual > 0:
    error = result["total"] - manual
    st.write(
        f"Difference vs manual: **{error:+d}** cells "
        f"({100.0 * abs(error) / manual:.1f} % error)"
    )


# ==========================================================
# SAVE TO VALIDATION LOG
# ==========================================================

if mode == "pollen":
    mc1, mc2 = st.columns(2)
    manual_live = mc1.number_input(
        "Manual LIVE count (optional)", min_value=0, value=0, step=1
    )
    manual_dead = mc2.number_input(
        "Manual DEAD count (optional)", min_value=0, value=0, step=1
    )
else:
    manual_live = manual_dead = 0

image_name = st.text_input(
    "Image name for the log", value=getattr(file, "name", "image")
)

if st.button("Save this result to the validation log"):
    try:
        save_to_log({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image": image_name,
            "cell_type": mode,
            "app_total": result["total"],
            "app_live": result["live"] if mode == "pollen" else "",
            "app_dead": result["dead"] if mode == "pollen" else "",
            "manual_total": manual if manual > 0 else "",
            "manual_live": manual_live if (manual_live + manual_dead) > 0 else "",
            "manual_dead": manual_dead if (manual_live + manual_dead) > 0 else "",
            "method": method,
            "radius_min": r_lo,
            "radius_max": r_hi,
            "sensitivity": sensitivity,
            "hue_lo": hue_lo if (mode == "pollen" and colour_rule == "hsv") else "",
            "hue_hi": hue_hi if (mode == "pollen" and colour_rule == "hsv") else "",
            "sat_min": sat_min if (mode == "pollen" and colour_rule == "hsv") else "",
            "colour_rule": colour_rule if mode == "pollen" else "",
            "blue_margin": blue_margin if (mode == "pollen" and colour_rule == "blueness") else "",
            "sq_left": left,
            "sq_top": top,
            "sq_width": width,
            "sq_height": height
        })
        st.success(f"Saved to {LOG_FILE.name}")
    except PermissionError:
        st.error(
            f"Could not write {LOG_FILE.name}. Close it in Excel "
            "(or any other program) and press the button again."
        )


# ==========================================================
# EXPORT
# ==========================================================

st.subheader("3. Export")

df = pd.DataFrame(result["objects"])

if not df.empty:
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download per-cell table (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="cells.csv",
        mime="text/csv"
    )

summary = pd.DataFrame([{
    "cell_type": mode,
    "total": result["total"],
    "live": result["live"] if mode == "pollen" else "",
    "dead": result["dead"] if mode == "pollen" else "",
    "viability_percent": round(result["viability"], 2) if mode == "pollen" else "",
    "dilution": dilution,
    "squares": squares,
    "cells_per_mL": round(concentration),
    "manual_count": manual if manual > 0 else ""
}])

st.download_button(
    "Download summary (CSV)",
    summary.to_csv(index=False).encode("utf-8"),
    file_name="summary.csv",
    mime="text/csv"
)

ok, png = cv2.imencode(".png", result["annotated"])
if ok:
    st.download_button(
        "Download annotated image (PNG)",
        png.tobytes(),
        file_name="annotated.png",
        mime="image/png"
    )