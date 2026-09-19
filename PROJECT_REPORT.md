# Heimdall — DepthWizard: Technical Report

**Single-view height estimation and 3D flythrough** · SIH 2026 · Problem Statement 26175 (ISRO / SAC)

---

## Executive summary

Heimdall converts a **single optical RGB remote-sensing image** into an elevation model and a navigable 3D scene.

* **Georeferenced input (GeoTIFF)** → **absolute DSM** in metres, in the image's own CRS. It combines a terrain datum from a coarse DEM (Copernicus GLO-30 fetched automatically, or a user-supplied SRTM / CartoDEM) with a learned **height-above-ground (nDSM)** map, and can optionally be corrected with ground control points.
* **Non-georeferenced input (PNG / JPG)** → **relative DSM (rDSM)**: height above local ground, in metres when the ground sample distance is known or can be estimated.
* **Visualisation**: a web platform (Next.js + React Three Fiber) with upload, live progress, a UV-textured terrain mesh, orbit and first-person flight, true-metre height/slope readouts, distance and profile measurement, water-level simulation, and built-in **validation against reference DSM / LiDAR rasters**.

The height model is a frozen **Depth Anything V3 Metric-Large** backbone plus a trained **ASPP domain-adaptation head**. The head predicts per-pixel scale and shift fields that map backbone depth to metric height above ground. It was trained on GAMUS (0.33 m RGB + nDSM).

---

## 1. Problem-statement mapping

| Requirement (PS 26175) | Implementation |
|---|---|
| Pre-trained monocular backbone for relative depth | Depth Anything V3 Metric-Large, frozen (`heimdall/depth`, `depth_anything_3/`) |
| rDSM for non-georeferenced imagery | `infer.py` on PNG/JPG → `*_rdsm.tif` (+ PNG, mesh) |
| Absolute DSM for georeferenced imagery using SRTM-class DEM or GCPs | DTM from reprojected coarse DEM (auto Copernicus GLO-30 / user DEM) + nDSM; GCP offset/plane correction (`heimdall/calibration/dem.py`) |
| Output DSM in a standard geospatial format | Float32 GeoTIFF with source CRS and geotransform, nodata, units and product tags |
| Project the optical image onto a terrain mesh | Grid mesh with UVs onto the full-resolution image, exported as glTF binary (`heimdall/mesh`) |
| Rendering engine, first-person navigation, heights/slopes from any viewpoint | Three.js / R3F viewer: orbit, fly (terrain collision), hover readouts, slope layer, measure/profile, flood level |
| Upload imagery, visualise, validate against references | Web app job API + validation panel (`POST /api/jobs/:id/validate`) |
| RMSE / MAE / correlation, stability across landscapes | `validate.py`, `eval.py`: RMSE, MAE, bias, NMAD, LE90, Pearson r, R², per landscape / class / city |
| Standalone deployment | `run.sh`, Next.js standalone build, Dockerfile with baked model weights (offline) |

---

## 2. Pipeline

```
[optical image]
   │ Stage 1  Ingest — GeoTIFF (CRS, transform, nodata) or plain image; band order; 16-bit percentile stretch
   │ Stage 2  GSD — from the geotransform (projected or geographic CRS), --gsd, or a vehicle-size heuristic
   │ Stage 3  Resample to the working GSD (0.33 m = training GSD; up-sampling capped at 3×)
   │ Stage 4  Height — DA3 Metric-L (frozen) + ASPP head on 512² tiles, raised-cosine blending → nDSM (m)
   │ Stage 5  Calibration — DSM = DTM + nDSM; DTM from coarse DEM reprojected to the image grid,
   │          morphological opening (bare earth) + smooth up-sampling; optional GCP plane
   │ Stage 6  Outputs — DSM / nDSM / DTM GeoTIFF, 16-bit PNG, hill-shaded preview, metadata JSON
   ▼ Stage 7  Mesh — UV-textured GLB (true metres, Y-up), height grid for exact viewer readouts
```

Modules: `heimdall/pipeline.py` orchestrates everything and is shared by the CLI, the web server and the evaluators. Progress is emitted as structured JSON events, which the web app streams to the browser.

### 2.1 Why resample to the training GSD
The head learned *metric* heights from GAMUS imagery at 0.33 m/px. A monocular model infers height partly from apparent object size, so feeding it 0.01 m drone imagery or 1 m satellite imagery at native resolution changes what "a building" looks like. Heimdall therefore resamples every georeferenced image (and any image with a known GSD) to 0.33 m before inference, then writes the DSM on that working grid with a consistent geotransform.

### 2.2 Height model
* **Backbone**: Depth Anything V3 Metric-Large (~334 M parameters, frozen). DA3 returns *depth*, where larger means farther. For the backbone-only path this is sign-corrected to a relative height (height ≈ H_sensor − depth for a distant nadir sensor).
* **Head** (`heimdall/decoder/head.py`): RGB + backbone depth → conv stem → ASPP (dilations 1/6/12/18 + global pooling) → two maps, a scale `S(x,y)` and a shift `T(x,y)`:

  `nDSM(x, y) = ReLU( S(x, y) · D(x, y) + T(x, y) )`

  Initialised to the identity (scale bias 1, shift bias 0). 1.37 M trainable parameters.
* **Inference** uses the same 512² tile size as training, 128 px overlaps with raised-cosine blending, and optional flip test-time augmentation.

### 2.3 Object-based regularisation (`heimdall/refine`)
Per-pixel monocular prediction leaks image texture into height (e.g. solar-panel grids become bumps), and its limited resolution turns walls into ramps several metres wide. After inference, the image is segmented into colour-homogeneous regions (Felzenszwalb, Lab), regions whose predicted heights are clearly bimodal are split at the Otsu threshold (so a grey roof and an adjacent grey road separate, with the wall at the ramp's mid-height), and thin mixed-colour edge strips are absorbed into their neighbours. Then:
* built regions (≥ 1.5 m, not vegetated) → one flat level, or a pitched plane only if it explains the data much better;
* adjacent built regions within 0.5 m → merged to one level;
* vegetation → smoothed prediction kept (canopies aren't planar);
* ground → one level per region;
* **vehicle ground prior**: vehicles are detected with a DOTA-trained oriented-box YOLO (aerial imagery). Surfaces with vehicles parked on them are levelled with the surrounding ground, because large paved areas such as parking lots are otherwise easily mistaken for flat concrete roofs. Vehicles become ~1.4 m boxes.

It runs in ~0.5 s per 300² grid, is on by default (`--no-refine` to disable), and its effect on accuracy is measured with `eval.py --refine`.

### 2.4 Absolute calibration
Single-image height models recover heights *relative to the local ground*; absolute elevation needs a datum.

1. **Coarse DEM** — user file or Copernicus GLO-30 COGs streamed from AWS Open Data (only the blocks covering the scene; cached). The DEM is reprojected onto a ~30 m grid over the image footprint in the image CRS.
2. **Bare-earth filtering** — 30 m radar DEMs partially contain buildings and canopy. A grey-scale opening (3×3 cells) removes positive bumps smaller than ~90 m, followed by light smoothing and cubic-spline up-sampling to the working grid → DTM.
3. **DSM = DTM + nDSM**.
4. **GCPs** (optional; `lon,lat,z`, `x,y,z` or `row,col,z`): one or two points apply a median offset, three or more fit a robust plane (iterative outlier rejection). The report includes RMSE at the GCPs before and after correction.

Backbone-only mode fits the relative map to the DEM with RANSAC. It only does so when the DEM can constrain the fit (at least 8 DEM cells across the scene and at least 5 m of relief) and the fitted scale is positive; otherwise it outputs a unitless rDSM and says so.

### 2.5 Outputs
`*_dsm.tif` (or `*_rdsm.tif`), `*_ndsm.tif`, `*_dtm.tif`, `*_height16.png` (metre offset/scale stored in PNG text chunks), `*_preview.png`, `*_texture.jpg`, `*_mesh.glb`, `*_grid.bin`, `*_meta.json` (product, units, GSD source, CRS, centre lat/lon, calibration details, statistics, warnings, timings).

---

## 3. Loss and training

```
Total = 1.0 · SILog + 0.5 · GradientMatching(4 scales) + 0.1 · L1
```

* **SILog** (Eigen et al.) on `log1p` heights: `log1p` keeps 0 m ground finite; the variance is clamped at 1e-8 before the square root.
* **Multi-scale gradient matching** (steps 1, 2, 4, 8 px) keeps roof edges and terrain breaks sharp.
* **L1 anchor** keeps the absolute scale from drifting.

**Training run used for the shipped checkpoint** (Kaggle, 2× Tesla T4): 524 GAMUS tiles, 512² random crops, batch 2 with 4-step gradient accumulation, AdamW (lr 1e-4, wd 1e-4), cosine schedule, AMP, 78 completed epochs in a 12-hour session. `decoder_best_all.pth` is the epoch-77 checkpoint (lowest training loss, 12.80).

**Training script improvements** (`scripts/train_decoder.py`) for further runs:
* flip / 90° rotation / photometric augmentation;
* a validation split evaluated at native GSD (centre crops, not resizing);
* best checkpoint selected by **validation RMSE**;
* `--resume` for fine-tuning;
* DA3 run one image at a time (batched calls make DA3 treat unrelated tiles as views of one scene; measured 2.3 % change in depth).

---

## 4. Evaluation methodology

### 4.1 Metrics
All metrics are computed over **every valid pixel, ground included**:

| Metric | Definition |
|---|---|
| RMSE | √mean(e²), e = pred − ref |
| MAE | mean(|e|) |
| Bias | mean(e) |
| NMAD | 1.4826 · median(|e − median(e)|) — robust spread |
| LE90 | 90th percentile of |e| |
| Pearson r, R² | linear agreement / explained variance |
| δ < 1.25ⁿ | ratio accuracy on pixels with ref ≥ 1 m |

### 4.2 Stratification
* **Landscape (per tile)**: urban (≥ 12 % building pixels), forested (≥ 20 % tall vegetation), sparse (< 5 % objects), mixed. Hilly scenes are identified by terrain relief when an absolute reference is available.
* **Pixel class**: building (≥ 2.5 m, not vegetated), tree (≥ 2.5 m, vegetated by excess-green index), ground, low vegetation; plus *steep terrain* (> 8° smoothed slope) for absolute DSM validation.
* **City** (GAMUS tile prefix) and a **predict-zero baseline** for context.

### 4.3 Validation against a reference DSM / LiDAR (`validate.py`, web panel)
The reference is reprojected onto the prediction grid, using average resampling when it is finer. Three sets of metrics are reported:
* **raw**;
* **after removing the median vertical offset**, because datum differences such as ellipsoidal LiDAR vs the EGM2008 geoid are constant shifts unrelated to the height model;
* **above-ground**: both surfaces are reduced to heights above a morphological ground estimate. This is the fair comparison for rDSM products.

Also written: an error map, a scatter plot, an error histogram, and per-class tables. A synthetic check (reference = prediction + 30 m + N(0, 0.5 m) noise, reprojected to EPSG:4326) recovers the 30 m offset exactly, with a 0.26 m residual RMSE.

### 4.4 Results

> **To be filled from `eval.py` output.** Run on the GAMUS **test** split (not used in training):
>
> ```bash
> python eval.py --weights checkpoints/decoder/decoder_best_all.pth --data-dir <GAMUS> --split test \
>                --output results/gamus_test.json
> ```
>
> Copy the OVERALL, landscape and pixel-class rows and the predict-zero baseline here. Do not quote numbers that were not produced by this command.

| Group | RMSE (m) | MAE (m) | Bias (m) | NMAD (m) | r |
|---|---|---|---|---|---|
| Overall | | | | | |
| Predict-zero baseline | | | | | |
| Urban | | | | | |
| Forested | | | | | |
| Sparse | | | | | |
| Building pixels | | | | | |
| Tree pixels | | | | | |
| Ground pixels | | | | | |

### 4.5 Qualitative end-to-end run (bundled demo)
Georeferenced drone orthomosaic, Kathmandu (EPSG:32645, 1.25 cm GSD, 7649 × 8154 px):
* resampled to 0.33 m (290 × 309 working grid);
* terrain datum from Copernicus GLO-30, fetched automatically: 1307–1309 m;
* resulting DSM: 1307.6–1316.6 m, with above-ground heights up to about 9 m;
* total runtime about 13 s on an Apple-silicon laptop (MPS), including the DEM download.

Structure is correct (roads and courtyards low, roofs and tree crowns high). Multi-storey buildings appear under-estimated, which the GAMUS evaluation will quantify.

---

## 5. Visualisation platform

* **Upload** a GeoTIFF / PNG / JPG, with an optional DEM and GCP CSV and advanced options (GSD override, band order, TTA, auto-DEM, model mode).
* **Live progress**: stage list, per-tile progress bar, warnings and log.
* **Result panel**: product type, statistics, GSD source, CRS, centre coordinates, calibration method and source, and downloads of every output.
* **3D viewer**:
  * UV-textured mesh with the full-resolution optical image;
  * layers: optical, elevation, height above ground, slope in degrees;
  * sun-direction relief shading;
  * vertical exaggeration slider (readouts always report true metres);
  * orbit and first-person flight (WASD, speed scaled to the scene, terrain collision);
  * hover readout of elevation, above-ground height, slope, map coordinates and lat/lon;
  * two-point measurement with an elevation profile;
  * flood water level with inundated fraction and area;
  * screenshots.
* **Validation panel**: upload a reference raster to get metric tables, per-class errors, a scatter plot, a histogram and an error map.
* **2D maps** and **metadata** tabs, and a recent-jobs list.

## 6. Deployment

* `./run.sh`: first run creates the virtualenv, installs dependencies, downloads models and builds the web app; it then serves on port 3000.
* **Docker**: the multi-stage image bakes the DA3 weights, YOLO and the decoder head into the image, so it runs offline (`--build-arg TORCH_INDEX=…cu121` for CUDA).
* **Offline operation**: `scripts/prefetch_models.py` then `HF_HUB_OFFLINE=1`. Without internet, supply `--reference-dem` or GCPs for absolute heights.
* **Tests**: `pytest -q tests` covers tiling, polarity, band order / 16-bit stretch, GSD, DEM reprojection (metric values preserved), GCP plane fit, RANSAC, metrics, datum-offset validation and mesh export.

---

## 7. Anticipated reviewer questions

**Q1. Why freeze the Depth Anything backbone?**
The labelled set is small relative to the 334 M-parameter backbone; full fine-tuning risks catastrophic forgetting of the general depth priors. A light head (1.4 M parameters) learns the domain mapping from backbone depth to metric above-ground height.

**Q2. Why predict scale and shift fields instead of height directly?**
The backbone already encodes relative geometry. A spatially varying affine map is a strong, well-conditioned prior, and it is initialised to the identity so training starts from the backbone's structure.

**Q3. How do you get absolute elevation from one image?**
You can't from pixels alone. The model recovers height above ground, and absolute elevation comes from a datum: a 30 m DEM (reprojected and bare-earth filtered) or GCPs. The DSM is their sum, and the metadata records which datum was used.

**Q4. Why resample to 0.33 m?**
That is the GSD the head was trained at. Monocular height cues are scale-dependent, so matching the training GSD is the single most important factor for metric accuracy across sensors.

**Q5. How do you validate against LiDAR with a different datum?**
The validator reports raw metrics, offset-removed metrics (a constant datum shift is not a model error) and above-ground metrics, all on the reprojected reference grid, and it writes an error map.

**Q6. Why SILog with `log1p`?**
Ground pixels in an nDSM are exactly 0 m; `log1p` stays finite there. The variance clamp prevents infinite gradients near zero error. Gradient matching sharpens edges, and L1 anchors the absolute scale.

**Q7. Latency?**
Measured on an Apple-silicon laptop (MPS): about 1.3 s per 512² tile. The 7649 × 8154 demo GeoTIFF takes about 13 s end to end, including the DEM download. CUDA GPUs are faster.

**Q8. Known weaknesses?**
* Trained on 524 GAMUS tiles (urban US/EU, 0.33 m): coverage of Indian urban morphology, forests and hilly terrain is limited.
* Absolute accuracy is bounded by the 30 m datum.
* Heights of tall buildings tend to be under-estimated.

---

## 8. Roadmap

1. **Train on the full GAMUS training split** (6,304 tiles vs the 524 used), with augmentation and validation-based model selection.
2. Fine-tune on Indian imagery (e.g. Cartosat with CartoDEM / LiDAR where available) at the evaluation GSD.
3. Use DA3's `depth_conf` to down-weight unreliable pixels in the loss and export a confidence layer.
4. ONNX / TensorRT export for edge deployment.
