/*
 * core.js - the counting logic that does NOT need OpenCV or the page.
 * It is a line-by-line port of cell_detection.py + viability.py so the
 * phone app gives the same results as the Streamlit app.
 *
 * Works in the browser (window.Core) and in Node (module.exports) so it can
 * be tested.
 */
(function (root) {
  "use strict";

  const Core = {};

  // ---------- counting square ----------

  Core.clampSquare = function (sq, W, H, minSize) {
    minSize = minSize || 20;
    let x = Math.trunc(sq[0]), y = Math.trunc(sq[1]);
    let w = Math.trunc(sq[2]), h = Math.trunc(sq[3]);
    x = Math.max(0, Math.min(x, W - minSize));
    y = Math.max(0, Math.min(y, H - minSize));
    w = Math.max(minSize, Math.min(w, W - x));
    h = Math.max(minSize, Math.min(h, H - y));
    return [x, y, w, h];
  };

  Core.squareFromPercent = function (W, H, left, top, width, height) {
    return Core.clampSquare(
      [W * left / 100, H * top / 100, W * width / 100, H * height / 100],
      W, H
    );
  };

  // ---------- small helpers ----------

  function median(values) {
    const a = Float64Array.from(values);
    a.sort();
    const n = a.length;
    if (n === 0) return 0;
    return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
  }

  // ---------- colour measurements (RGBA pixel data, like canvas ImageData) ----------

  // Blueness of the slide background: median of 100*ln(B/R) over the brighter
  // half of the pixels (where there are no cells).
  Core.backgroundBlueness = function (rgba, W, H) {
    const n = W * H;
    const bright = new Float64Array(n);
    const logRatio = new Float64Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const R = rgba[p], G = rgba[p + 1], B = rgba[p + 2];
      bright[i] = (R + G + B) / 3;
      logRatio[i] = 100 * Math.log((B + 1) / (R + 1));
    }
    const med = median(bright);
    const sel = [];
    for (let i = 0; i < n; i++) if (bright[i] >= med) sel.push(logRatio[i]);
    return median(sel);
  };

  // 100*ln(meanB / meanR) inside the inner 70 % of a circle. Mirrors viability.py:
  // the circle is shrunk by one pixel (3x3 cross erosion) so the blurred cell
  // edge is ignored.
  Core.cellBlueness = function (rgba, W, H, cx, cy, r) {
    const ccx = Math.trunc(cx), ccy = Math.trunc(cy);
    const R0 = Math.trunc(0.7 * r);
    const R2 = R0 * R0;

    // bounding box of the circle, clipped to the image (like the Python patch)
    const bx0 = Math.max(0, ccx - R0), bx1 = Math.min(W - 1, ccx + R0);
    const by0 = Math.max(0, ccy - R0), by1 = Math.min(H - 1, ccy + R0);

    function inside(x, y) {
      if (x < bx0 || x > bx1 || y < by0 || y > by1) return true;   // border: not eroded
      const dx = x - ccx, dy = y - ccy;
      return dx * dx + dy * dy <= R2;
    }

    function mean(erode) {
      let sumB = 0, sumR = 0, count = 0;
      for (let y = by0; y <= by1; y++) {
        for (let x = bx0; x <= bx1; x++) {
          const dx = x - ccx, dy = y - ccy;
          if (dx * dx + dy * dy > R2) continue;
          if (erode && !(inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1))) continue;
          const p = (y * W + x) * 4;
          sumR += rgba[p];
          sumB += rgba[p + 2];
          count++;
        }
      }
      return count ? { b: sumB / count, r: sumR / count } : null;
    }

    const m = mean(true) || mean(false);
    if (!m) return 0;
    return 100 * Math.log((m.b + 1) / (m.r + 1));
  };

  // ---------- circle clean-up ----------

  // Hough returns the strongest circles first; drop circles that overlap a
  // stronger one.
  Core.dedupeCircles = function (circles) {
    const kept = [];
    for (const c of circles) {
      let ok = true;
      for (const k of kept) {
        if (Math.hypot(c.x - k.x, c.y - k.y) < 0.7 * Math.max(c.r, k.r)) { ok = false; break; }
      }
      if (ok) kept.push(c);
    }
    return kept;
  };

  // Hemocytometer border rule: cells touching the RIGHT or BOTTOM edge of the
  // counting square are ignored (left / top are counted).
  Core.touchesRightBottom = function (c, W, H) {
    return c.x + c.r > W - 1 || c.y + c.r > H - 1;
  };

  // ---------- the whole counting step after circle detection ----------
  /*
   * circles : [{x, y, r}] in crop coordinates (as returned by Hough)
   * rgba    : crop pixel data, W x H
   * opts    : { mode: "rbc" | "pollen", margin: number }
   */
  Core.process = function (circles, rgba, W, H, opts) {
    const mode = opts.mode || "rbc";
    const margin = opts.margin == null ? 10 : opts.margin;

    const kept = Core.dedupeCircles(circles);
    const stainRule = mode === "pollen";
    const bgRef = stainRule ? Core.backgroundBlueness(rgba, W, H) : 0;

    const objects = [];
    const excluded = [];
    let live = 0, dead = 0;

    for (const c of kept) {
      if (Core.touchesRightBottom(c, W, H)) {
        excluded.push(c);
        continue;
      }
      let label = "live";
      let blue = 0;
      if (stainRule) {
        blue = Core.cellBlueness(rgba, W, H, c.x, c.y, c.r) - bgRef;
        label = blue >= margin ? "dead" : "live";
      }
      if (label === "dead") dead++; else live++;
      objects.push({ id: objects.length + 1, x: c.x, y: c.y, r: c.r, blueness: blue, label: label });
    }

    const total = live + dead;
    return {
      objects: objects,
      excluded: excluded,
      live: live,
      dead: dead,
      total: total,
      viability: total ? 100 * live / total : 0,
      bgRef: bgRef
    };
  };

  // cells/mL = (count / squares) x dilution x 10^4  (1 mm x 1 mm x 0.1 mm = 0.1 uL)
  Core.concentration = function (total, squares, dilution) {
    return (total / Math.max(squares, 1)) * dilution * 1e4;
  };

  Core.toCSV = function (rows, columns) {
    const esc = v => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const lines = [columns.join(",")];
    for (const r of rows) lines.push(columns.map(c => esc(r[c])).join(","));
    return lines.join("\n") + "\n";
  };

  if (typeof module !== "undefined" && module.exports) module.exports = Core;
  else root.Core = Core;

})(typeof self !== "undefined" ? self : this);
