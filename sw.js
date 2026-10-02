/* Service worker: caches the app so it works with NO internet after the first visit. */
const CACHE = "cellcounter-v1";
const FILES = [
  "./", "index.html", "style.css", "core.js", "app.js",
  "opencv.js", "manifest.webmanifest",
  "icons/icon-192.png", "icons/icon-512.png"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (cache) {
      // add one by one so a single missing file does not break the install
      return Promise.all(FILES.map(function (f) {
        return cache.add(f).catch(function () { /* skip missing file */ });
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; })
                             .map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request).then(function (hit) {
      return hit || fetch(e.request).then(function (resp) {
        const copy = resp.clone();
        caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
        return resp;
      });
    })
  );
});
