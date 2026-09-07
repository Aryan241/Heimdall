# Heimdall 

Heimdall is an end-to-end software suite that transforms **single-view optical RGB remote-sensing images** into high-precision, metric elevation maps (DSMs) and projects them into a fully navigable, interactive 3D flythrough environment.

---

## 🎯 Solution Architecture

This project is built to explicitly solve the two core deliverables of the hackathon:

### 1. Elevation Estimation Module (Python Backend)
A robust command-line inference engine (`infer.py`) powered by a fine-tuned Depth Anything V2 backbone. 
- **Input Adaptability:** Natively ingests both non-georeferenced imagery (JPG, PNG) and spatially aware formats (GeoTIFF) using `rasterio`.
- **Absolute Scale Calibration (Fine-Tuned):** Features a custom `DomainAdaptationWrapper` that can load fine-tuned decoder weights to predict **absolute metric heights** directly from the image, completely bypassing heuristic calibration!
- **Geospatial Export:** Automatically outputs a high-fidelity Digital Surface Model (DSM). If a GeoTIFF is provided, the output is a single-band **GeoTIFF** preserving the original Coordinate Reference System (CRS). It also exports a `.ply` 3D mesh for web rendering.

### 2. Interactive Visualization Platform (Next.js Web UI)
A user-friendly, standalone web application (`heimdall-web/`) designed for seamless evaluation and interaction.
- **Integrated Next.js API:** Direct browser-to-backend communication via a dedicated API route, streamlining file uploads and mesh generation.
- **1-Click Processing:** Users can drag-and-drop satellite imagery directly into the browser to trigger the Python processing pipeline on the backend.
- **Cinematic 3D Flythrough:** Once meshed, the UI dynamically loads the terrain using WebGL (Three.js), enabling a 360-degree interactive orbital flythrough of the projected structural heights.
- **Enhanced WebGL Resolution:** Generates dense, full-resolution 3D meshes (up to 1M vertices) with a 2x Z-exaggeration to dramatically highlight structural visibility.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- PyTorch (with MPS/CUDA support recommended for fast inference)

### 1. Install Backend Dependencies
```bash
cd Heimdall
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
cd heimdall-web
npm install
npm run dev
```

### 3. Usage
1. Open your browser and navigate to `http://localhost:3000`.
2. Click **"Drop Satellite Image Here"** to upload an optical image (JPG, PNG, TIFF).
3. Click **"Process Pipeline"**.
4. The system will upload the image, run the PyTorch elevation extraction, generate a dense 3D mesh, and instantly load it into the interactive viewer for you to navigate!

---

## 🛠 Tech Stack
- **AI / Computer Vision:** PyTorch, Depth Anything V2, Hugging Face Transformers
- **Geospatial Processing:** Rasterio, Trimesh, NumPy
- **Frontend / Visualization:** Next.js, React, Three.js (`@react-three/fiber`), Tailwind CSS

---

*This project was developed for the SIH 2026 Hackathon to bridge the domain gap between natural egocentric depth models and top-down remote sensing elevation requirements.*
