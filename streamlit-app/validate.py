"""
Validation of the app against manual counts.

Usage (in the same folder as validation_log.csv):
    python validate.py
    python validate.py my_other_log.csv

Outputs
-------
validation_summary.csv                 one row of metrics per cell type
validation_<type>_scatter.png          app count vs manual count
validation_<type>_bland_altman.png     agreement plot
(pollen only, if manual live/dead were entered)
validation_pollen_viability.png        app vs manual viability %
"""

import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG = sys.argv[1] if len(sys.argv) > 1 else "validation_log.csv"


def scatter_plot(manual, app, title, path, unit="cells"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(manual, app, s=40)

    lim = max(manual.max(), app.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x (perfect)")

    if len(manual) >= 2 and np.ptp(manual) > 0:
        slope, intercept = np.polyfit(manual, app, 1)
        xs = np.array([0, lim])
        ax.plot(xs, slope * xs + intercept, "r-", linewidth=1,
                label=f"fit: y = {slope:.2f}x + {intercept:.1f}")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Manual ({unit})")
    ax.set_ylabel(f"App ({unit})")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def bland_altman(manual, app, title, path):
    mean = (manual + app) / 2.0
    diff = app - manual
    bias = diff.mean()
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(mean, diff, s=40)
    ax.axhline(bias, color="r", label=f"bias = {bias:+.1f}")
    ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--",
               label=f"+1.96 SD = {bias + 1.96 * sd:+.1f}")
    ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--",
               label=f"-1.96 SD = {bias - 1.96 * sd:+.1f}")
    ax.set_xlabel("Mean of app and manual")
    ax.set_ylabel("App - manual")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return bias, sd


def main():
    try:
        df = pd.read_csv(LOG)
    except FileNotFoundError:
        sys.exit(f"Could not find {LOG}. Save some results from the app first.")

    df["manual_total"] = pd.to_numeric(df["manual_total"], errors="coerce")
    df["app_total"] = pd.to_numeric(df["app_total"], errors="coerce")
    df = df.dropna(subset=["manual_total", "app_total"])
    df = df[df["manual_total"] > 0]

    # keep the latest entry per image if you saved one twice
    df = df.drop_duplicates(subset=["image", "cell_type"], keep="last")

    if df.empty:
        sys.exit("No rows with a manual count yet. Enter a manual count in "
                 "the app before saving.")

    rows = []

    for cell_type, g in df.groupby("cell_type"):

        g = g.copy()
        manual = g["manual_total"].to_numpy(dtype=float)
        app = g["app_total"].to_numpy(dtype=float)
        n = len(g)

        err = app - manual
        mae = np.abs(err).mean()
        mape = (np.abs(err) / manual).mean() * 100
        bias = err.mean()

        if n >= 3 and manual.std() > 0 and app.std() > 0:
            r = np.corrcoef(manual, app)[0, 1]
        else:
            r = float("nan")

        scatter_plot(manual, app,
                     f"{cell_type.upper()}: app vs manual (n={n})",
                     f"validation_{cell_type}_scatter.png")
        _, sd = bland_altman(manual, app,
                             f"{cell_type.upper()}: Bland-Altman",
                             f"validation_{cell_type}_bland_altman.png")

        row = {
            "cell_type": cell_type,
            "n_images": n,
            "MAE_cells": round(mae, 2),
            "MAPE_percent": round(mape, 2),
            "bias_cells": round(bias, 2),
            "SD_of_difference": round(sd, 2),
            "pearson_r": round(r, 3) if not np.isnan(r) else "",
            "R_squared": round(r * r, 3) if not np.isnan(r) else ""
        }

        # Pollen viability (needs manual live and dead)
        if cell_type == "pollen":
            for c in ["app_live", "app_dead", "manual_live", "manual_dead"]:
                g[c] = pd.to_numeric(g[c], errors="coerce")

            # A blank live or dead box means 0 when the other one is filled in
            only_live = g["manual_live"].notna() & g["manual_dead"].isna()
            only_dead = g["manual_dead"].notna() & g["manual_live"].isna()
            g.loc[only_live, "manual_dead"] = (
                g["manual_total"] - g["manual_live"]).clip(lower=0)
            g.loc[only_dead, "manual_live"] = (
                g["manual_total"] - g["manual_dead"]).clip(lower=0)

            v = g.dropna(subset=["app_live", "app_dead",
                                 "manual_live", "manual_dead"])
            v = v[(v["manual_live"] + v["manual_dead"]) > 0]
            v = v[(v["app_live"] + v["app_dead"]) > 0]

            if len(v) >= 2:
                app_v = 100 * v["app_live"] / (v["app_live"] + v["app_dead"])
                man_v = 100 * v["manual_live"] / (v["manual_live"] + v["manual_dead"])
                row["viability_MAE_points"] = round(
                    float(np.abs(app_v - man_v).mean()), 2)
                row["viability_n"] = len(v)

                scatter_plot(man_v.to_numpy(), app_v.to_numpy(),
                             f"Pollen viability % (n={len(v)})",
                             "validation_pollen_viability.png",
                             unit="viability %")

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv("validation_summary.csv", index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved: validation_summary.csv and validation_*.png")
    print("Tip: aim for at least 30 images per cell type for your report.")


if __name__ == "__main__":
    main()