"""
Merge all validation_log*.csv files in this folder into ONE validation_log.csv.

Run once (in the same folder as app.py):
    python merge_logs.py

* Rows from every log are combined (older logs get empty new columns).
* Exact duplicate rows are removed.
* The original files are MOVED (not deleted) into a folder called log_backup.
"""

import csv
import shutil
from pathlib import Path

FOLDER = Path(__file__).parent

COLUMNS = [
    "timestamp", "image", "cell_type",
    "app_total", "app_live", "app_dead",
    "manual_total", "manual_live", "manual_dead",
    "method", "radius_min", "radius_max", "sensitivity",
    "hue_lo", "hue_hi", "sat_min",
    "colour_rule", "blue_margin",
    "sq_left", "sq_top", "sq_width", "sq_height"
]


def main():
    files = sorted(FOLDER.glob("validation_log*.csv"))

    if not files:
        print("No validation_log*.csv files found in", FOLDER)
        return

    rows = []
    seen = set()

    for path in files:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r = {(k or "").strip(): v for k, v in r.items()}
                clean = {k: r.get(k, "") for k in COLUMNS}

                key = tuple(clean[k] for k in COLUMNS)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(clean)

        print(f"Read {path.name}")

    rows.sort(key=lambda r: r["timestamp"])

    backup = FOLDER / "log_backup"
    backup.mkdir(exist_ok=True)
    for path in files:
        shutil.move(str(path), str(backup / path.name))

    with open(FOLDER / "validation_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMerged {len(rows)} rows into validation_log.csv")
    print(f"Originals moved to {backup}")


if __name__ == "__main__":
    main()
