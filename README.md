# Cell Counter (phone / offline web app)

Counts RBC (total) and pollen (live / dead, trypan blue) from a hemocytometer
photo. Runs entirely on the device, works with no internet after the first
visit, and installs on a phone like an app (PWA). Photos are never uploaded.

It uses the same counting rules as the Streamlit app:
Hough-circle detection -> border rule (left/top counted, right/bottom ignored)
-> pollen live/dead by "relative blueness" -> concentration
= count / squares x dilution x 10^4.

## Files

| File | Purpose |
|---|---|
| `index.html`, `style.css` | the page |
| `app.js` | photo input, counting square, OpenCV.js detection, results, saving, CSV |
| `core.js` | counting + colour logic (port of `cell_detection.py` and `viability.py`) |
| `sw.js`, `manifest.webmanifest`, `icons/` | offline caching and "install as app" |
| `opencv.js` | **you must add this file** (see below) |

## Setup (once)

1. Download OpenCV.js (about 8-10 MB), e.g. from
   https://docs.opencv.org/4.10.0/opencv.js
   and save it in this folder as `opencv.js` (next to `index.html`).
2. Test on your computer:
   ```
   cd cellcounter-web
   python -m http.server 8000
   ```
   Open http://localhost:8000 in Chrome. The badge at the top should change from
   "Loading OpenCV..." to **"Ready (offline)"**.
3. Choose a photo, move the green box, press **Count cells**.

## Put it on a phone (free, HTTPS)

Offline install needs HTTPS, so host the folder on GitHub Pages:

1. Create a GitHub repository and upload everything in this folder
   (including `opencv.js`).
2. Settings -> Pages -> Deploy from branch -> `main` / root.
3. Open `https://<your-user>.github.io/<repo>/` on the phone in Chrome
   **with internet**, wait for "Ready (offline)".
4. Chrome menu -> **Add to Home screen / Install app**.
5. Turn on airplane mode and open the app from its icon. It should still work.

## Notes

- "Take photo" opens the phone camera. "Choose from gallery" picks a saved photo.
- Saved results stay on the device (Save result). Export all (CSV) downloads them.
- Detection starts with the same defaults as the Streamlit app
  (RBC radius 8-16 px / sensitivity 18, pollen 24-44 px / 26, blueness margin 10).
  Photos are shrunk to 1600 px, like in the Streamlit app, so radii mean the same thing.
- After you change any file and re-upload, increase `CACHE` in `sw.js`
  (for example `cellcounter-v2`) so phones pick up the new version.
- Research and educational use only. Not a diagnostic device.
