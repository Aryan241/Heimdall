# Heimdall web app

Next.js 16 (App Router) + React Three Fiber front end for the Heimdall pipeline.

## Run

```bash
npm ci
npm run build && npm start      # production (http://localhost:3000)
npm run dev                     # development
```

The server spawns the Python pipeline in the parent folder. Override locations with
`HEIMDALL_ROOT`, `HEIMDALL_PYTHON` and `HEIMDALL_JOBS_DIR` (default `../outputs/jobs`).
`output: "standalone"` is enabled; the Docker image in the repo root runs `.next/standalone/server.js`.

## API

| Route | Purpose |
|---|---|
| `POST /api/jobs` | multipart `image` (+ `reference_dem`, `gcps`, `options` JSON). Streams SSE: `job`, `event` (pipeline stages/progress/warnings), `log`, `done`, `error`. |
| `GET /api/jobs` | recent jobs |
| `GET /api/jobs/:id/files/:name` | job outputs (GeoTIFF, GLB, PNG, JSON, height grid) |
| `POST /api/jobs/:id/validate` | multipart `reference` DSM/LiDAR GeoTIFF → metrics, scatter, histogram, error map |

## Viewer

* **Orbit / Fly / Measure** modes. Fly is first-person (WASD, Space/E up, Q/C down, Shift boost), with speed scaled to the scene and terrain collision.
* **Layers**: optical texture, elevation, height above ground, slope (degrees), all with sun-direction relief shading.
* **Hover readout** of elevation, above-ground height, slope, map coordinates (raster CRS) and approximate lat/lon, sampled from the exported height grid (true metres, independent of the vertical-exaggeration slider).
* **Measure**: horizontal and 3-D distance, height difference, grade and an elevation profile.
* **Flood**: water plane at a chosen level, with the inundated fraction and area.
* **Screenshot** export.

`public/demo/` holds a bundled demo scene, generated with `infer.py --name demo`, that loads on first visit.
