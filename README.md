# Heimdall — DepthWizard

**Single-view height estimation and 3D flythrough** (SIH 2026 · PS 26175 · ISRO / SAC)

Heimdall turns one optical RGB remote-sensing image into an elevation model and an interactive 3D scene:

| Input | Output |
|---|---|
| **GeoTIFF** (CRS + geotransform) | **Absolute DSM** GeoTIFF in metres (same CRS): terrain datum from a coarse DEM (Copernicus GLO-30 fetched automatically, or your SRTM / CartoDEM) + predicted above-ground height, optionally corrected with GCPs |
| **PNG / JPG / TIFF** (no georeferencing) | **Relative DSM (rDSM)**: height above local ground, in metres when the GSD is known (or estimated) |

Every run also writes an nDSM (height above ground), the terrain datum, a 16-bit PNG, a hill-shaded preview, a **UV-textured GLB mesh** and a metadata JSON. The web app uploads imagery, streams progress, renders the textured terrain for orbit / first-person flight, reads out true heights and slopes, measures distances and profiles, simulates water levels, and **validates the result against a reference DSM / LiDAR raster**.

---

## How it works

```
image ─► ingest (bands, 16-bit stretch, nodata, CRS)
      ─► GSD (geotransform │ --gsd │ vehicle-size heuristic)
      ─► resample to 0.33 m/px (the training GSD of the decoder head)
      ─► Depth Anything V3 Metric-Large (frozen) + trained ASPP head, 512² tiles, feathered blending
            → nDSM: height above ground (m)
      ─► object-based regularisation: colour segments → flat / pitched roofs, vertical walls;
            detected vehicles mark ground (parking lots stay flat)            (--no-refine to skip)
      ─► calibration
            georeferenced: DSM = DTM(reference DEM, reprojected + bare-earth filtered) + nDSM  [+ GCP plane]
            plain image:   rDSM = nDSM                                                         [+ row/col GCPs]
      ─► GeoTIFF / PNG / preview / textured GLB / height grid / metadata
```

* **Backbone**: Depth Anything V3 Metric-Large, frozen (`depth_anything_3/`, weights from Hugging Face).
* **Head**: ASPP regression head (dilations 1/6/12/18) predicting per-pixel scale and shift of the backbone depth → metric height above ground. Trained on **GAMUS** (0.33 m RGB + nDSM) with SILog + multi-scale gradient-matching + L1 loss (`scripts/train_decoder.py`). Checkpoint: `checkpoints/decoder/decoder_best_all.pth`.
* **Why resample to 0.33 m**: the head learned heights at the GAMUS ground sample distance; running it at another GSD changes the apparent size of every object. Up-sampling is capped at 3× for coarse imagery.
* **Absolute scale**: the coarse DEM is reprojected onto the image grid (any CRS), filtered with a morphological opening to suppress buildings/canopy that leak into 30 m radar DEMs, and smoothly up-sampled. GCPs (CSV) remove the residual offset or tilt.
* **Backbone-only mode** (`--weights none`): relative height from DA3 (depth sign-corrected), RANSAC-calibrated to a DEM when the DEM can actually constrain it (enough cells and relief), otherwise output as a unitless rDSM.

---

## Quick start

```bash
./run.sh            # first run: creates venv, installs deps, downloads models, builds the web app
                    # then serves http://localhost:3000
```

Manual setup:

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision            # see INSTALL_PYTORCH.md for CUDA builds
pip install -r requirements.txt
python scripts/prefetch_models.py        # DA3 + YOLO weights (then HF_HUB_OFFLINE=1 works)

cd heimdall-web && npm ci && npm run build && npm start
```

Docker (bakes all model weights in; runs offline):

```bash
docker build -t heimdall .                                                       # CPU
docker build -t heimdall --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .   # CUDA
docker run --rm -p 3000:3000 --gpus all -v "$PWD/outputs:/app/outputs" heimdall
```

Environment variables for the web server: `HEIMDALL_ROOT` (repo path), `HEIMDALL_PYTHON` (interpreter), `HEIMDALL_JOBS_DIR` (default `outputs/jobs`), `HEIMDALL_CACHE` (DEM cache).

---

## Command line

```bash
# GeoTIFF → absolute DSM (Copernicus GLO-30 terrain fetched automatically)
python infer.py -i scene.tif -o outputs/

# Your own DEM and ground control points
python infer.py -i scene.tif --reference-dem srtm_30m.tif --gcps gcps.csv

# Plain image with known GSD; multispectral product stored as B,G,R,NIR
python infer.py -i image.jpg --gsd 0.5
python infer.py -i cartosat_mx.tif --bands 3,2,1

# Evaluate with and without roof/wall regularisation to measure its effect
#   python eval.py ... --refine

# Higher quality (flip test-time augmentation, ~2× slower)
python infer.py -i scene.tif --tta
```

GCP CSV headers: `lon,lat,z` (WGS84), `x,y,z` (image CRS) or `row,col,z` (pixels on the working grid).

Outputs (`<name>_*`): `dsm.tif` / `rdsm.tif`, `ndsm.tif`, `dtm.tif`, `height16.png` (metre scale stored in PNG text chunks), `preview.png`, `texture.jpg`, `mesh.glb`, `grid.bin`, `meta.json`.

## Validation and benchmarking

```bash
# A result against a reference DSM / LiDAR raster (any CRS; reprojected automatically)
python validate.py --meta outputs/scene_meta.json --reference lidar_dsm.tif

# The decoder head on the GAMUS test split, at native resolution with production tiling
python eval.py --weights checkpoints/decoder/decoder_best_all.pth --data-dir /path/to/GAMUS --split test \
               --output results/gamus_test.json            # add --every 4 for a quick run
```

Reported metrics: RMSE, MAE, bias, NMAD, LE90, Pearson r, R², δ-accuracy. They are computed over **all** valid pixels (ground included), overall, per landscape (urban / forested / sparse / mixed), per pixel class (building / tree / ground / low vegetation; plus steep terrain for absolute DSMs), per city, and against a predict-zero baseline. `validate.py` reports raw metrics, metrics after removing the vertical-datum offset, and above-ground (nDSM-vs-nDSM) metrics, and writes an error map.

## Training

```bash
python scripts/train_decoder.py --dataset-dir /path/to/GAMUS --epochs 40 --batch-size 4
# fine-tune the shipped head
python scripts/train_decoder.py --dataset-dir /path/to/GAMUS --resume checkpoints/decoder/decoder_best_all.pth --epochs 10 --lr 2e-5
```

Flip / 90° rotation / photometric augmentation, a validation split at native GSD, and best-checkpoint selection by validation RMSE are built in.

## Tests

```bash
pytest -q tests
```

## Repository layout

```
heimdall/               pipeline package
  pipeline.py           end-to-end orchestration (used by CLI, web app, evaluators)
  ingestion/            loading (bands, 16-bit, nodata) and tiling (feathered blending)
  depth/                Depth Anything V2/V3 wrappers
  decoder/              ASPP head, loss, dataset, inference
  calibration/          DEM fetch/reprojection, GCPs, RANSAC, GSD heuristic
  mesh/ output/ eval/   textured GLB, writers, metrics + DSM comparison
depth_anything_3/       bundled Depth Anything V3 code
infer.py validate.py eval.py
scripts/                training, model prefetch
heimdall-web/           Next.js + React Three Fiber web app (see heimdall-web/README.md)
tests/                  unit tests
```

## Known limitations

* The head predicts height above ground learned from GAMUS (US/European cities, 0.33 m). Accuracy on other sensors, regions and GSDs should be measured with `eval.py` / `validate.py` before quoting numbers.
* Absolute accuracy of the terrain datum is bounded by the coarse DEM (Copernicus GLO-30 is EGM2008-referenced; reference LiDAR is often ellipsoidal, which is why the validator reports an offset-removed score).
* Automatic DEM download needs internet access; offline, supply `--reference-dem` or GCPs.
