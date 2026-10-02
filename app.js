/*
 * app.js - the page: photo input, counting square, OpenCV.js circle detection,
 * results, saved history and CSV export. All counting rules live in core.js.
 */
(function () {
  "use strict";

  const $ = id => document.getElementById(id);

  const MAX_SIDE = 1600;                 // phone photos are shrunk to this size (same as the Streamlit app)
  const DEFAULTS = {                     // same starting values as cell_detection.py / viability.py
    rbc:    { minR: 8,  maxR: 16, sens: 18 },
    pollen: { minR: 24, maxR: 44, sens: 26 }
  };
  const STORE_KEY = "cellcounter_history_v1";

  let mode = "rbc";
  let cvReady = false;
  let hasImage = false;
  let lastResult = null;
  const srcCanvas = document.createElement("canvas");   // full (resized) photo

  // ---------------------------------------------------------------
  // OpenCV.js loading (works with the classic and the Promise build)
  // ---------------------------------------------------------------
  function markReady() {
    cvReady = true;
    $("status").textContent = "Ready (offline)";
    $("status").className = "badge ok";
  }

  function waitForCv() {
    const t0 = Date.now();
    (function poll() {
      if (typeof cv !== "undefined" && cv) {
        if (cv.Mat) { markReady(); return; }
        if (typeof cv.then === "function") {
          cv.then(function (m) { window.cv = m; markReady(); });   // callback returns nothing on purpose
          return;
        }
        cv.onRuntimeInitialized = markReady;
        return;
      }
      if (Date.now() - t0 > 90000) {
        $("status").textContent = "OpenCV did not load";
        $("status").className = "badge bad";
        return;
      }
      setTimeout(poll, 150);
    })();
  }

  // ---------------------------------------------------------------
  // small UI helpers
  // ---------------------------------------------------------------
  function msg(text) { $("msg").textContent = text; }

  function bindRange(id, outId) {
    const el = $(id), out = $(outId);
    const upd = () => { out.textContent = el.value; };
    el.addEventListener("input", upd);
    upd();
  }

  function currentSquare() {
    return Core.squareFromPercent(
      srcCanvas.width, srcCanvas.height,
      +$("left").value, +$("top").value, +$("width").value, +$("height").value
    );
  }

  function drawView() {
    if (!hasImage) return;
    const c = $("view");
    c.width = srcCanvas.width;
    c.height = srcCanvas.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(srcCanvas, 0, 0);
    const [x, y, w, h] = currentSquare();
    ctx.lineWidth = Math.max(3, c.width / 250);
    ctx.strokeStyle = "#00e676";
    ctx.strokeRect(x, y, w, h);
  }

  function applyMode(newMode) {
    mode = newMode;
    const d = DEFAULTS[mode];
    $("minR").value = d.minR; $("maxR").value = d.maxR; $("sens").value = d.sens;
    ["minR", "maxR", "sens"].forEach(id => $(id).dispatchEvent(new Event("input")));
    document.querySelectorAll(".pollenOnly").forEach(el => { el.hidden = mode !== "pollen"; });
    $("results").hidden = true;
    lastResult = null;
  }

  // ---------------------------------------------------------------
  // photo loading
  // ---------------------------------------------------------------
  async function loadFile(file) {
    if (!file) return;
    msg("Loading photo…");
    let bmp;
    try {
      bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch (e) {
      try { bmp = await createImageBitmap(file); }
      catch (e2) { msg("Could not read this photo."); return; }
    }
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    srcCanvas.width = Math.round(bmp.width * scale);
    srcCanvas.height = Math.round(bmp.height * scale);
    srcCanvas.getContext("2d").drawImage(bmp, 0, 0, srcCanvas.width, srcCanvas.height);
    if (bmp.close) bmp.close();

    hasImage = true;
    $("viewSection").hidden = false;
    $("results").hidden = true;
    drawView();
    msg("Photo loaded. Position the green box, then press Count cells.");
    $("viewSection").scrollIntoView({ behavior: "smooth" });
  }

  // ---------------------------------------------------------------
  // detection (OpenCV.js) + counting (core.js)
  // ---------------------------------------------------------------
  function houghCircles(imageData, minR, maxR, sens) {
    const src = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    const circles = new cv.Mat();
    try {
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, gray, new cv.Size(7, 7), 0, 0, cv.BORDER_DEFAULT);
      const minDist = Math.max(8, Math.trunc(1.6 * minR));
      cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1.2, minDist, 60, sens,
                      Math.trunc(minR), Math.trunc(maxR));
      const d = circles.data32F, out = [];
      for (let i = 0; i < circles.cols; i++) {
        out.push({ x: d[i * 3], y: d[i * 3 + 1], r: d[i * 3 + 2] });
      }
      return out;
    } finally {
      src.delete(); gray.delete(); circles.delete();
    }
  }

  function runCount() {
    if (!hasImage) { msg("Take or choose a photo first."); return; }
    if (!cvReady) { msg("OpenCV is still loading, try again in a moment."); return; }

    const btn = $("btnCount");
    btn.disabled = true; btn.textContent = "Counting…";

    // let the button repaint before the heavy work
    setTimeout(function () {
      try {
        const [x, y, w, h] = currentSquare();
        const crop = srcCanvas.getContext("2d").getImageData(x, y, w, h);
        const minR = +$("minR").value, maxR = +$("maxR").value, sens = +$("sens").value;

        const circles = houghCircles(crop, minR, Math.max(maxR, minR + 1), sens);
        const res = Core.process(circles, crop.data, w, h, { mode: mode, margin: +$("margin").value });

        render(res, crop, [x, y, w, h], { minR: minR, maxR: maxR, sens: sens });
      } catch (e) {
        msg("Counting failed: " + (e && e.message ? e.message : e));
      } finally {
        btn.disabled = false; btn.textContent = "Count cells";
      }
    }, 30);
  }

  function render(res, crop, square, params) {
    const out = $("out");
    out.width = crop.width; out.height = crop.height;
    const ctx = out.getContext("2d");
    ctx.putImageData(crop, 0, 0);

    const lw = Math.max(2, out.width / 300);
    function ring(c, colour, width) {
      ctx.beginPath(); ctx.arc(c.x, c.y, c.r, 0, 2 * Math.PI);
      ctx.lineWidth = width; ctx.strokeStyle = colour; ctx.stroke();
    }
    res.excluded.forEach(c => ring(c, "#ffeb3b", Math.max(1, lw / 2)));
    res.objects.forEach(o => ring(o, o.label === "dead" ? "#ff1744" : "#00e676", lw));

    $("mTotal").textContent = res.total;
    $("mLive").textContent = res.live;
    $("mDead").textContent = res.dead;
    $("mViab").textContent = res.viability.toFixed(1) + "%";

    const squares = Math.max(1, +$("squares").value || 1);
    const dilution = Math.max(1, +$("dilution").value || 1);
    const conc = Core.concentration(res.total, squares, dilution);
    $("conc").textContent = "Concentration: " + Math.round(conc).toLocaleString() +
      " cells/mL (count / squares x dilution x 10^4). Ignored at right/bottom border: " + res.excluded.length + ".";

    lastResult = { res: res, square: square, params: params, conc: conc, squares: squares, dilution: dilution };
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth" });
  }

  // ---------------------------------------------------------------
  // saved results + CSV export (stored on this device only)
  // ---------------------------------------------------------------
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); }
    catch (e) { return []; }
  }
  function storeHistory(list) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(list)); return true; }
    catch (e) { msg("Could not save on this device (storage blocked)."); return false; }
  }

  function renderHistory() {
    const list = loadHistory();
    $("historySection").hidden = list.length === 0;
    const box = $("history");
    box.textContent = "";
    list.slice().reverse().forEach(function (r) {
      const d = document.createElement("div");
      d.className = "item";
      d.textContent = r.time + " | " + r.cell_type + " | total " + r.total +
        (r.cell_type === "pollen" ? " (live " + r.live + ", dead " + r.dead + ", " + r.viability_percent + "%)" : "") +
        (r.manual_count !== "" ? " | manual " + r.manual_count : "") +
        " | " + r.cells_per_mL + " cells/mL";
      box.appendChild(d);
    });
  }

  function saveResult() {
    if (!lastResult) return;
    const r = lastResult, res = r.res;
    const manual = $("manual").value;
    const list = loadHistory();
    list.push({
      time: new Date().toISOString().slice(0, 19).replace("T", " "),
      cell_type: mode,
      total: res.total,
      live: mode === "pollen" ? res.live : "",
      dead: mode === "pollen" ? res.dead : "",
      viability_percent: mode === "pollen" ? res.viability.toFixed(1) : "",
      dilution: r.dilution,
      squares: r.squares,
      cells_per_mL: Math.round(r.conc),
      manual_count: manual === "" ? "" : +manual,
      sq_x: r.square[0], sq_y: r.square[1], sq_w: r.square[2], sq_h: r.square[3],
      min_radius: r.params.minR, max_radius: r.params.maxR, sensitivity: r.params.sens,
      blue_margin: mode === "pollen" ? +$("margin").value : ""
    });
    if (storeHistory(list)) { renderHistory(); msg("Saved. Use Export all (CSV) to download every saved result."); }
  }

  function exportCSV() {
    const list = loadHistory();
    if (!list.length) { msg("Nothing saved yet."); return; }
    const cols = ["time", "cell_type", "total", "live", "dead", "viability_percent", "dilution", "squares",
                  "cells_per_mL", "manual_count", "sq_x", "sq_y", "sq_w", "sq_h",
                  "min_radius", "max_radius", "sensitivity", "blue_margin"];
    const blob = new Blob([Core.toCSV(list, cols)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "cell_counts.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ---------------------------------------------------------------
  // wiring
  // ---------------------------------------------------------------
  function init() {
    [["left", "oLeft"], ["top", "oTop"], ["width", "oWidth"], ["height", "oHeight"],
     ["minR", "oMinR"], ["maxR", "oMaxR"], ["sens", "oSens"], ["margin", "oMargin"]]
      .forEach(p => bindRange(p[0], p[1]));

    ["left", "top", "width", "height"].forEach(id => $(id).addEventListener("input", drawView));

    document.querySelectorAll("input[name=mode]").forEach(el =>
      el.addEventListener("change", () => { if (el.checked) applyMode(el.value); }));

    $("btnCamera").addEventListener("click", () => $("fileCamera").click());
    $("btnGallery").addEventListener("click", () => $("fileGallery").click());
    $("fileCamera").addEventListener("change", e => { loadFile(e.target.files[0]); e.target.value = ""; });
    $("fileGallery").addEventListener("change", e => { loadFile(e.target.files[0]); e.target.value = ""; });

    $("btnCount").addEventListener("click", runCount);
    $("btnSave").addEventListener("click", saveResult);
    $("btnExport").addEventListener("click", exportCSV);
    $("btnClear").addEventListener("click", function () {
      if (confirm("Delete all saved results from this device?")) { storeHistory([]); renderHistory(); }
    });

    applyMode("rbc");
    renderHistory();
    waitForCv();

    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("sw.js").catch(function () { /* works without it, just not offline */ });
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
