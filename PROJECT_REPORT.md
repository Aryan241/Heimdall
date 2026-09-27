# Heimdall — DepthWizard: Technical Report

**Single-view height estimation and 3D flythrough** · SIH 2026 · Problem Statement 26175 (ISRO / SAC)

---

## Executive summary

Heimdall converts a **single optical RGB remote-sensing image** into an elevation model and a navigable 3D scene.

* **Georeferenced input (GeoTIFF)** → **absolute DSM** in metres, in the image's own CRS. It combines a terrain datum from a coarse DEM (Copernicus GLO-30 fetched automatically, or a user-supplied SRTM / CartoDEM) with a learned **height-above-ground (nDSM)** map, and can optionally be corrected with ground control points.
* **Non-georeferenced input (PNG / JPG)** → **relative DSM (rDSM)**: height above local ground, in metres when the ground sample distance is known or can be estimated.
* **Visualisation**: a web platform (Next.js + React Three Fiber) with upload, live progress, a UV-textured terrain mesh, orbit and first-person flight, true-metre height/slope readouts, distance and profile measurement, water-level simulation, and built-in **validation against reference DSM / LiDAR rasters**.

The height model is a frozen **Depth Anything V2 Base** backbone plus a trained **multi-scale decoder** that consumes the DPT feature pyramid and predicts per-pixel scale and shift fields mapping backbone depth to metric height above ground. Trained on 9 004 tiles (GAMUS + Dutch AHN LiDAR at two ground resolutions); scored against airborne LiDAR at eight held-out Dutch sites at **5.68 m RMSE / r = 0.66**, versus 8.37 m for predicting zero and 7.44 m for the previous checkpoint (§4.4).

---

## 1. Problem-statement mapping

| Requirement (PS 26175) | Implementation |
|---|---|
| Pre-trained monocular backbone for relative depth | Depth Anything V2 (frozen); V3 bundled and selectable (`heimdall/depth`, `depth_anything_3/`) |
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
   │ Stage 4  Height — DA2 (frozen) + multi-scale decoder on 512² tiles, raised-cosine blending → nDSM (m)
   │ Stage 5  Calibration — DSM = DTM + nDSM; DTM from coarse DEM reprojected to the image grid,
   │          morphological opening (bare earth) + smooth up-sampling; optional GCP plane
   │ Stage 6  Outputs — DSM / nDSM / DTM GeoTIFF, 16-bit PNG, hill-shaded preview, metadata JSON
   ▼ Stage 7  Mesh — UV-textured GLB (true metres, Y-up), height grid for exact viewer readouts
```

Modules: `heimdall/pipeline.py` orchestrates everything and is shared by the CLI, the web server and the evaluators. Progress is emitted as structured JSON events, which the web app streams to the browser.

### 2.1 Why resample to the training GSD
The head learned *metric* heights from GAMUS imagery at 0.33 m/px. A monocular model infers height partly from apparent object size, so feeding it 0.01 m drone imagery or 1 m satellite imagery at native resolution changes what "a building" looks like. Heimdall therefore resamples every georeferenced image (and any image with a known GSD) to 0.33 m before inference, then writes the DSM on that working grid with a consistent geotransform.

### 2.2 Height model
* **Backbone**: Depth Anything V2 Base (~98 M parameters, frozen), selected by measured correlation with LiDAR height (§4.4). It returns relative *disparity* (larger = closer = taller). Depth Anything V3's *monocular* model scores the same and is selectable (`--model da3-mono-l`, needs a head trained for it); V3's *metric* model is unusable for nadir imagery (§4.4).
* **Head** (`heimdall/decoder/multiscale_head.py`): the backbone's DPT neck exposes a
  four-stage feature pyramid (128 channels each, at strides ~1/14 to ~1/1.75) *before* it is
  collapsed into a single depth number. The decoder consumes that pyramid, fuses it
  coarse-to-fine with DPT-style residual blocks, applies dilated context (1/6/12/18 plus a
  global pool) so it can distinguish a tall building from a bright roof at ground level, and
  refines at full resolution conditioned on the RGB image and the backbone depth. It outputs
  two maps, a scale `S(x,y)` and a shift `T(x,y)`:

  `nDSM(x, y) = ReLU( S(x, y) · D(x, y) + T(x, y) )`

  Initialised to the identity (scale bias 1, shift bias 0), so training starts from "height =
  backbone depth" and learns the correction. 1.72 M trainable parameters; GroupNorm throughout,
  so small batches behave like large ones.

  The earlier head (`heimdall/decoder/head.py`, ASPP over RGB + the scalar depth map) is kept
  for loading legacy checkpoints and is selectable with `--head affine`. It sees only the
  collapsed depth number, so everything the transformer knew about roof planes and façades had
  already been discarded before it ran — which is why the pyramid version replaced it.
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

### 4.4 Results — independent LiDAR benchmark (Netherlands, AHN)

`scripts/benchmark_ahn.py` downloads open Dutch aerial orthophotos and AHN LiDAR (DSM and
DTM, 0.5 m), runs the production pipeline and scores predicted height above ground against
`AHN DSM − AHN DTM`. This is a genuinely independent test: a different country, sensor and
vendor from the GAMUS training data, and every site is excluded by a 3 km buffer from the
AHN tiles used for training (`scripts/build_ahn_dataset.py`). 400 m sites, 0.25 m imagery.

| Site | Landscape | RMSE (m) | MAE (m) | Bias (m) | r | Predict-zero RMSE (m) |
|---|---|---|---|---|---|---|
| amsterdam_centre | dense historic urban | 8.52 | 5.74 | −5.28 | 0.86 | 13.32 |
| rotterdam_centre | high-rise urban | 16.94 | 8.90 | −6.55 | 0.61 | 22.92 |
| utrecht_suburb | suburban residential | 2.49 | 1.56 | −0.91 | 0.78 | 4.75 |
| rotterdam_port | industrial | 8.10 | 5.02 | −4.72 | 0.84 | 11.01 |
| westland_greenhouses | greenhouses | 2.22 | 1.64 | −0.54 | 0.63 | 3.90 |
| veluwe_forest | forest | 2.26 | 1.13 | −0.41 | 0.79 | 4.25 |
| flevoland_farmland | sparse rural | 1.53 | 0.23 | +0.03 | 0.07 | 1.43 |
| valkenburg_hills | hilly town | 3.41 | 1.85 | −1.03 | 0.71 | 5.38 |
| **mean** | | **5.68** | **3.26** | — | **0.662** | **8.37** |

**Against the previous (shipped) head**, same sites, same script:

| | mean RMSE | mean MAE | mean r |
|---|---|---|---|
| Shipped head (DA3-metric backbone) | 7.44 m | 4.63 m | 0.377 |
| **Retrained head (DA2 + multiscale decoder)** | **5.68 m** | **3.26 m** | **0.662** |
| Predict 0 m everywhere | 8.37 m | — | — |

Seven of the eight sites improved; the greenhouses are unchanged (2.20 → 2.22 m), which is
expected — they are near-flat structures where there was little error to remove. The
correlation gain is the more meaningful result: 0.377 → 0.662 means the model now tracks the
*shape* of the built environment, not just its average height. Amsterdam's historic centre
moves from r = −0.08 (anti-correlated, i.e. worse than useless) to 0.86.

The model beats the predict-zero baseline at seven of eight sites. The exception is
Flevoland farmland (1.53 m vs 1.43 m), which is reclaimed polder with almost no
above-ground structure — predicting zero there is close to optimal and the model's small
positive bias costs it slightly.

Object-based regularisation ("refined") again changes the metrics by < 0.03 m. It is kept
because it improves the appearance of roofs in the 3D view, and this confirms it does so
without distorting the heights.

**Where the remaining error is.** Two places, consistently across both benchmarks:

* **Tall buildings.** Rotterdam's high-rise core is the worst site (16.94 m) and carries a
  −6.55 m bias; on GAMUS, building pixels score 7.27 m RMSE with −2.47 m bias. A single view
  gives no parallax, so a 60 m tower and a 30 m one differ mainly in shadow length and
  perspective cues that an orthomosaic has deliberately removed.
* **Tree canopy.** GAMUS tree pixels are the worst class (6.85 m RMSE, −3.98 m bias, NMAD
  4.73 m). Canopy height varies over metres within a crown, and the backbone sees texture
  rather than structure.

Both are under-predictions, which is the safe direction for most downstream uses (a flood or
line-of-sight model that under-calls a building is conservative) but is the obvious target
for further work: height-weighted loss terms, or shadow-length features as an explicit input.

**Root cause of the original failure (measured).** Correlation between each frozen backbone's output and LiDAR
height, same eight sites, 0.33 m tiles:

| Backbone | Type | Mean r | Median r | s/tile (MPS) |
|---|---|---|---|---|
| DA2 Small | relative | 0.35 | — | 1.5 |
| **DA2 Base (default)** | relative | **0.41** | **0.56** | **1.6** |
| DA2 Large | relative | 0.41 | — | 6.6 |
| DA3 Mono-Large | relative | 0.40 | 0.58 | 4.6 |
| DA3-Large | general / multi-view | 0.23 | 0.29 | 6.1 |
| DA3 Metric-Large (shipped head's backbone) | metric | 0.06 | — | 26 |

The split is **relative vs metric**, not V2 vs V3. Metric models regress absolute distance
and therefore assume a pinhole camera with a known focal length (`apply_metric_scaling` in
`depth_anything_3/utils/alignment.py`); an orthorectified mosaic has no single camera and no
perspective, so the metric branch degenerates to a near-constant plane (measured output range
0.6–0.7 m across a whole city tile). Relative models only have to rank near vs far, which in a
nadir view is exactly "taller vs shorter"; the metres then come from the trained head and the
DEM datum. DA3-Large is built for multi-view input and gets only a single tile here.

DA2 Base and DA3 Mono are statistically indistinguishable on this test, so the default is DA2
Base for being ~3× faster. DA3 Mono is selectable (`--model da3-mono-l`) but needs a head
trained against it — the head is fitted to one backbone's output distribution. Two sites
(flat farmland, water-dominated port) have almost no above-ground structure, so their
correlations are noise; the median column is the more meaningful summary.

The pipeline default is therefore Depth Anything V2 Base, and the head was retrained against
it from scratch (`docs/TRAINING.md`) — a head is fitted to one backbone's output
distribution, so the old weights could not be fine-tuned. The table at the top of this
section is that retrained model, measured by re-running the same benchmark.

### 4.5 Ground-truth comparison in the viewer

`scripts/build_demo_site.py` builds two scenes for the same place: Heimdall's reconstruction
from the orthophoto, and a second scene whose heights come from the AHN LiDAR instead of the
model. Both use the same texture, working grid, mesh resolution and vertical origin, so
switching between them in the web viewer isolates exactly what the model got wrong. The
reference scene reuses the pipeline's own terrain datum and replaces only the above-ground
layer, so the constant offset between the coarse DEM's geoid and the LiDAR's NAP datum
cancels out instead of contaminating the comparison.

Two pairs ship with the app, both from held-out benchmark sites:

| Scene | Landscape | RMSE | MAE | Bias | r | Predict-zero RMSE |
|---|---|---|---|---|---|---|
| `utrecht_suburb` | suburban residential, 600 m | 2.92 m | 1.89 m | −1.05 m | 0.769 | 5.39 m |
| `rotterdam_centre` | high-rise urban, 600 m | 14.20 m | 7.64 m | −5.73 m | 0.641 | 18.73 m |

Utrecht is the representative case and Rotterdam the hardest one in the whole benchmark; both
are shipped deliberately rather than only the flattering one. Flying the Rotterdam pair shows
the failure mode directly: the street layout, block structure and relative massing are right,
but the towers are built to roughly a third of their true height (model surface peaks at
37 m against the LiDAR's 115 m).

Any of the eight benchmark sites can be built the same way:

```bash
python scripts/build_demo_site.py --site amsterdam_centre --size 600
```

### 4.6 Qualitative end-to-end run (bundled drone scene)
Georeferenced drone orthomosaic, Kathmandu (EPSG:32645, 1.25 cm GSD, 7649 × 8154 px):
* resampled to 0.33 m (290 × 309 working grid);
* terrain datum from Copernicus GLO-30, fetched automatically: 1307–1309 m;
* resulting DSM: 1308.3–1311.9 m, with above-ground heights up to about 4.3 m;
* total runtime about 11 s on an Apple-silicon laptop (MPS), including the DEM download.

Structure is correct (roads and courtyards low, roofs and tree crowns high). No reference
exists for this scene, so it is a qualitative check only — the retrained model produces a
flatter surface here than the previous checkpoint did (4.3 m against 9.4 m peak height), and
without ground truth neither can be called correct. The quantitative claims in this report all
come from §4.4 and §4.5, where LiDAR truth exists.

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
