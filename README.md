# Heimdall 🏔️

**Single-View Height Estimation & 3D Flythrough** — SIH Problem Statement 26175

> Transform a single RGB remote-sensing image into an elevation map and navigable 3D terrain in the browser.

---

## Architecture

```
RGB Image ─→ [Ingestion & Routing] ─→ [Depth Anything V2 (frozen)] ─→ Relative Depth Map
                   │                                                          │
                   ├─ GeoTIFF? ──→ [Scale Calibration via SRTM/GLO-30] ──→ Absolute DSM
                   │                                                          │
                   └─ Plain?   ──→ [Heuristic Scale via Object Detection] ──→ Relative DSM
                                                                              │
                                                              [Mesh Generation + RGB Drape]
                                                                              │
                                                        [CesiumJS / Three.js Flythrough Viewer]
```

### Pipeline Stages

| Stage | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | `heimdall/ingestion/` | Input detection, routing, tiling | ✅ Done |
| 2 | `heimdall/depth/` | Depth Anything V2 inference (frozen backbone) | ✅ Done |
| 3 | `heimdall/decoder/` | Decoder-only fine-tuning on DFC2019/ISPRS | 🔲 Planned |
| 4 | `heimdall/segmentation/` | Semantic segmentation (ground/building/veg) | 🔲 Planned |
| 5 | `heimdall/calibration/` | RANSAC scale calibration vs. coarse DEM | 🔲 Planned |
| 6 | `heimdall/scale_heuristic/` | Object-based scale for non-georef images | 🔲 Planned |
| 7 | `heimdall/output/` | GeoTIFF / 16-bit PNG output | ✅ Done |
| 8 | `heimdall/mesh/` | Triangulated 3D mesh + RGB texture drape | 🔲 Planned |
| 9 | `frontend/` | CesiumJS + Three.js flythrough viewer | 🔲 Planned |
| 10 | `heimdall/evaluation/` | Stratified RMSE/MAE evaluation harness | 🔲 Planned |

## Quick Start

### 1. Install PyTorch (platform-specific)

See [INSTALL_PYTORCH.md](INSTALL_PYTORCH.md) for your platform.

```bash
# macOS (Apple Silicon):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Ubuntu (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run inference

```bash
# Basic — uses ViT-B, auto-detects device (MPS on Mac, CUDA on Linux)
python infer.py --input path/to/image.jpg --output-dir outputs/

# ViT-Large backbone for higher quality
python infer.py --input path/to/image.jpg --output-dir outputs/ --model vit-l

# GeoTIFF input — will also produce a GeoTIFF depth output
python infer.py --input path/to/scene.tif --output-dir outputs/

# Force CPU / increase verbosity
python infer.py --input path/to/image.jpg -o outputs/ --device cpu -v
```

### Outputs

| File | Description |
|------|-------------|
| `*_depth16.png` | 16-bit grayscale heightmap (0–65535) |
| `*_depth_vis.png` | Colorized depth visualization (inferno colormap) |
| `*_depth.npy` | Raw float32 depth array for downstream processing |
| `*_depth.tif` | GeoTIFF depth (georeferenced inputs only) |

## Project Structure

```
Heimdall/
├── infer.py                    # Main CLI entry point (Stage 1+2+7)
├── requirements.txt            # Platform-agnostic deps
├── INSTALL_PYTORCH.md          # Platform-specific PyTorch install
├── heimdall/
│   ├── device.py               # Device detection (CUDA/MPS/CPU)
│   ├── ingestion/
│   │   ├── loader.py           # GeoTIFF/plain image detection & loading
│   │   └── tiling.py           # Image tiling & stitching
│   ├── depth/
│   │   └── depth_anything.py   # Depth Anything V2 inference
│   ├── decoder/                # (Stage 3) Fine-tuned decoder head
│   ├── segmentation/           # (Stage 4) Semantic segmentation
│   ├── calibration/            # (Stage 5) Scale calibration
│   ├── scale_heuristic/        # (Stage 6) Object-based scaling
│   ├── output/
│   │   └── writers.py          # Depth output writers
│   ├── mesh/                   # (Stage 8) 3D mesh generation
│   └── evaluation/             # (Stage 10) Stratified evaluation
├── frontend/                   # (Stage 9) CesiumJS + Three.js viewer
│   ├── cesium/
│   ├── threejs/
│   └── shared/
├── configs/                    # Hydra/YAML configs
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── data/                       # (gitignored) Datasets
├── checkpoints/                # (gitignored) Model weights
└── outputs/                    # (gitignored) Pipeline outputs
```

## Hardware Requirements

- **Inference (Mac):** Apple Silicon with MPS — works with ViT-B at ~512×512. ViT-L needs ≥16GB unified memory.
- **Inference (Linux):** Any CUDA GPU with ≥4GB VRAM for ViT-B, ≥8GB for ViT-L.
- **Training (Stage 3):** Ubuntu workstation — parallel single-GPU jobs across GTX 1080 / RTX 3070.

## License

MIT
