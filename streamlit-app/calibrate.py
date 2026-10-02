"""
Find the best FIXED settings from your calibration images.

Why: if you adjust sliders on every image, the accuracy you report is
meaningless. This script tries many settings on ALL your calibration
images at once and tells you the single best set. You then lock that
set and validate on NEW images.

Before running
--------------
1. In the app, process ~8 images of one cell type. For each one:
   put the green box on the counting square, type your MANUAL counts,
   and press "Save this result to the validation log".
   (Pollen: also type the manual LIVE and DEAD counts.)
2. Run, in the same folder as app.py:

   python calibrate.py --type rbc    --images "C:\\path\\to\\your\\dataset"
   python calibrate.py --type pollen --images "C:\\path\\to\\your\\dataset"

--images is the folder that contains the photos (sub-folders are fine).
The script finds each photo by the file name stored in the log.

Output: a ranked table, calibration_results_<type>.csv, and the exact
lines to paste into cell_detection.py / viability.py.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

from cell_detection import detect_cells, DEFAULT_DETECTION
from roi import square_from_percent
from viability import DEFAULT_PARAMS

MAX_SIDE = 1600          # same shrink as app.py
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)      # works with any path
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


def read_calibration_rows(log_path, cell_type):
    rows = {}
    with open(log_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip(): v for k, v in r.items()}
            if r.get("cell_type") != cell_type:
                continue
            manual = num(r.get("manual_total"))
            sq = [num(r.get(k)) for k in
                  ("sq_left", "sq_top", "sq_width", "sq_height")]
            if not manual or manual <= 0 or any(v is None for v in sq):
                continue
            rows[r["image"]] = {          # latest row per image wins
                "manual_total": manual,
                "manual_live": num(r.get("manual_live")),
                "manual_dead": num(r.get("manual_dead")),
                "sq": sq,
            }
    return rows


def find_images(folder):
    found = {}
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in IMAGE_EXT:
            found.setdefault(p.name.lower(), []).append(p)
    return found


def metrics(app, manual):
    app = np.asarray(app, dtype=float)
    manual = np.asarray(manual, dtype=float)
    err = app - manual
    mape = float((np.abs(err) / manual).mean() * 100)
    mae = float(np.abs(err).mean())
    bias = float(err.mean())
    r = float(np.corrcoef(app, manual)[0, 1]) if len(app) >= 3 and app.std() > 0 and manual.std() > 0 else float("nan")
    return mape, mae, bias, r


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["rbc", "pollen"])
    ap.add_argument("--images", required=True, help="folder with the photos")
    ap.add_argument("--log", default="validation_log.csv")
    args = ap.parse_args()

    rows = read_calibration_rows(args.log, args.type)
    if not rows:
        sys.exit(
            "No usable rows found. Each row needs a manual total and the "
            "counting-square columns (sq_left ...). Save results with the "
            "newest app.py, and enter your manual count first."
        )

    files = find_images(args.images)

    data = []
    for name, info in rows.items():
        matches = files.get(Path(name).name.lower())
        if not matches:
            print(f"  skipped (photo not found in --images folder): {name}")
            continue
        path = matches[0]
        if len(matches) > 1:
            print(f"  WARNING: {len(matches)} files are named '{name}'. Using:\n"
                  f"           {path}\n"
                  "           If this is the wrong one, the counts will not match.")
        img = load_image(path)
        if img is None:
            print(f"  skipped (could not read): {name}")
            continue
        square = square_from_percent(img.shape, *info["sq"])
        data.append((name, img, square, info))

    if not data:
        sys.exit("None of the logged photos were found. Check --images.")

    print(f"\nCalibrating on {len(data)} image(s) of type '{args.type}'.")
    if len(data) < 6:
        print("  WARNING: fewer than 6 images. Results will be unreliable; "
              "use 8 or more.")

    # ---- search grid ----
    if args.type == "rbc":
        min_rs, max_rs = [4, 6, 8, 10], [14, 16, 18, 20, 24]
        senss = [8, 10, 12, 14, 16, 18, 20, 22, 24, 28]
    else:
        min_rs, max_rs = [16, 20, 24, 28], [40, 44, 50, 60]
        senss = [10, 14, 18, 22, 26, 30, 34, 40]

    manual_totals = [d[3]["manual_total"] for d in data]

    results = []
    combos = [c for c in itertools.product(min_rs, max_rs, senss) if c[0] < c[1]]
    print(f"Trying {len(combos)} detection settings ...")

    for min_r, max_r, sens in combos:
        counts = []
        for name, img, square, info in data:
            res = detect_cells(img, square, mode=args.type, method="circles",
                               min_radius=min_r, max_radius=max_r,
                               sensitivity=sens)
            counts.append(res["total"])
        mape, mae, bias, r = metrics(counts, manual_totals)
        wape = float(np.abs(np.array(counts, dtype=float) - np.array(manual_totals)).sum()
                     / np.sum(manual_totals) * 100)
        results.append({"min_r": min_r, "max_r": max_r, "sens": sens,
                        "WAPE_pct": round(wape, 2),
                        "MAPE_pct": round(mape, 2), "MAE": round(mae, 2),
                        "bias": round(bias, 2),
                        "pearson_r": "" if np.isnan(r) else round(r, 3)})

    results.sort(key=lambda r: (r["WAPE_pct"], abs(r["bias"])))

    out_csv = f"calibration_results_{args.type}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    cur = DEFAULT_DETECTION[args.type]
    total_manual = float(np.sum(manual_totals))
    print(f"\nTotal manual count across the calibration images: {total_manual:.0f}")
    if total_manual < 150:
        print("  NOTE: fewer than ~150 cells in total. Per-image % errors are "
              "dominated by +/-1 cell, so treat the ranking as a rough guide and "
              "add images (mix sparse AND dense ones).")

    print("\nTop 5 settings (lowest pooled error = sum|app-manual| / sum manual):")
    print(f"  {'min_r':>5} {'max_r':>5} {'sens':>5} {'WAPE%':>7} {'MAPE%':>7} {'MAE':>6} {'bias':>6} {'r':>6}")
    for r in results[:5]:
        print(f"  {r['min_r']:>5} {r['max_r']:>5} {r['sens']:>5} {r['WAPE_pct']:>7} {r['MAPE_pct']:>7} "
              f"{r['MAE']:>6} {r['bias']:>6} {str(r['pearson_r']):>6}")

    cur_row = next((r for r in results if (r["min_r"], r["max_r"], r["sens"]) ==
                    (cur["min_r"], cur["max_r"], cur["sens"])), None)
    if cur_row:
        print(f"\nYour current defaults ({cur['min_r']}, {cur['max_r']}, {cur['sens']}): "
              f"WAPE {cur_row['WAPE_pct']} %, MAPE {cur_row['MAPE_pct']} %, bias {cur_row['bias']}")

    best = results[0]
    print("\n=== PASTE into cell_detection.py (DEFAULT_DETECTION) ===")
    print(f'    "{args.type}": {{"min_r": {best["min_r"]}, "max_r": {best["max_r"]}, "sens": {best["sens"]}}},')

    # ---- edge-of-grid warning ----
    for label, value, options in (("min_r", best["min_r"], min_rs),
                                  ("max_r", best["max_r"], max_rs),
                                  ("sens", best["sens"], senss)):
        if value in (min(options), max(options)):
            print(f"  NOTE: best {label} = {value} is at the EDGE of the search range, "
                  "so an even better value may exist outside it.")

    # ---- per-image diagnosis with the best settings ----
    debug_dir = Path("calibration_debug")
    debug_dir.mkdir(exist_ok=True)

    print("\nPer-image check with the best settings "
          "(pictures saved in calibration_debug/):")
    print(f"  {'image':<26} {'manual':>6} {'app':>5} {'err%':>7}  {'box px (x,y,w,h)':<22} {'img size':<10} {'median r'}")
    for name, img, square, info in data:
        res = detect_cells(img, square, mode=args.type, method="circles",
                           min_radius=best["min_r"], max_radius=best["max_r"],
                           sensitivity=best["sens"])
        err = 100.0 * (res["total"] - info["manual_total"]) / info["manual_total"]
        med_r = (res["median_area"] / np.pi) ** 0.5 if res["median_area"] else 0
        h, w = img.shape[:2]
        print(f"  {name[:26]:<26} {info['manual_total']:>6.0f} {res['total']:>5} {err:>+6.0f}%  "
              f"{str(tuple(square)):<22} {w}x{h:<5} {med_r:>6.1f}")
        cv2.imwrite(str(debug_dir / f"{args.type}_{Path(name).stem}.png"),
                    res["annotated"])

    print("\nHow to read this: if 'app' is far below 'manual' on EVERY image, "
          "the problem is not the settings. Open the pictures in "
          "calibration_debug/ and compare with what you counted. Typical causes: "
          "the manual count covers a bigger area than the green box, the wrong "
          "photo was matched, or the cells are a very different size.")

    # ---- pollen: live / dead colour margin ----
    if args.type == "pollen":
        have = [d for d in data
                if d[3]["manual_live"] is not None and d[3]["manual_dead"] is not None]
        if len(have) >= 2:
            print(f"\nChoosing the blueness margin on {len(have)} image(s) with manual live/dead ...")
            sweep = []
            man_live_tot = sum(d[3]["manual_live"] for d in have)
            man_dead_tot = sum(d[3]["manual_dead"] for d in have)
            man_viab = 100 * man_live_tot / max(man_live_tot + man_dead_tot, 1)

            for margin in range(2, 31, 2):
                app_live_tot = app_dead_tot = 0
                dead_abs = 0.0
                for name, img, square, info in have:
                    res = detect_cells(
                        img, square, mode="pollen", method="circles",
                        min_radius=best["min_r"], max_radius=best["max_r"],
                        sensitivity=best["sens"],
                        viability_params={"rule": "blueness", "blue_margin": margin})
                    app_live_tot += res["live"]
                    app_dead_tot += res["dead"]
                    dead_abs += abs(res["dead"] - info["manual_dead"])
                dead_wape = 100 * dead_abs / max(man_dead_tot, 1)
                app_viab = 100 * app_live_tot / max(app_live_tot + app_dead_tot, 1)
                sweep.append((margin, dead_wape, app_viab, abs(app_viab - man_viab)))

            sweep.sort(key=lambda s_: (s_[1], s_[3]))
            print(f"  manual: {man_live_tot:.0f} live, {man_dead_tot:.0f} dead "
                  f"(viability {man_viab:.1f} %)")
            if man_dead_tot < 30 or man_live_tot < 30:
                print("  NOTE: fewer than ~30 live or ~30 dead grains in total. "
                      "A margin picked from so few grains can be wrong; keep the "
                      "default of 10 unless a larger set agrees.")
            print(f"  {'margin':>6} {'dead WAPE%':>11} {'app viab %':>11} {'|diff| pts':>11}")
            for m, d, v, dv in sweep[:5]:
                print(f"  {m:>6} {d:>11.1f} {v:>11.1f} {dv:>11.1f}")
            print("\n=== PASTE into viability.py (DEFAULT_PARAMS -> pollen) ===")
            print(f'        "blue_margin": {sweep[0][0]},')
            print("  (also set the slider default in app.py: 0, 60, <this value>)")
        else:
            print("\nThe blueness margin was not tuned: it needs at least 2 images "
                  "that have BOTH a manual live and a manual dead count "
                  f"(found {len(have)}). The default of 10 stays.")

    print(f"\nFull table saved to {out_csv}")
    print("NEXT: lock these settings, then validate on NEW images only.")


if __name__ == "__main__":
    main()