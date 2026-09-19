"""
Stage 3 — Unified Dataset Loader for Domain Adaptation

Loads RGB imagery and ground-truth height maps from the GAMUS dataset (HDF5 .h5)
and legacy formats (DFC2019 / ISPRS with .png/.tif files).
Handles random cropping, flip/rotation augmentation, and strata filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

logger = logging.getLogger(__name__)


def _load_h5_array(path: Path) -> np.ndarray:
    """Load a numpy array from an HDF5 (.h5) file.
    
    GAMUS stores each sample as a single dataset inside the .h5 file.
    We read the first dataset found.
    """
    import h5py
    with h5py.File(path, "r") as f:
        # GAMUS .h5 files typically have a single dataset at the root level.
        # Try common keys first, then fall back to the first key found.
        for key in ["data", "image", "rgb", "height", "agl", "dsm"]:
            if key in f:
                return np.array(f[key])
        # Fallback: grab the first dataset
        first_key = list(f.keys())[0]
        return np.array(f[first_key])


class RemoteSensingHeightDataset(Dataset):
    """Unified dataset for GAMUS (HDF5), DFC2019, and ISPRS Potsdam/Vaihingen.
    
    Supports directory structures:
        GAMUS:
            dataset_dir/
                images/train/  (*.h5 RGB files)
                heights/train/ (*.h5 AGL height files)
        DFC2019 / ISPRS:
            dataset_dir/
                images/   (RGB .jpg/.png/.tif)
                heights/  (Grayscale or float .png/.npy/.tif)
    """
    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".h5"}
    SUPPORTED_HEIGHT_EXTS = {".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".h5"}

    def __init__(
        self,
        dataset_dir: str | Path,
        patch_size: int = 512,
        split: Literal["train", "val"] = "train",
        strata: str | None = None,
        augment: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.patch_size = patch_size
        self.split = split
        self.strata = strata
        self.augment = augment
        
        # Try GAMUS structure first: images/{split}/ and heights/{split}/
        self.img_dir = self.dataset_dir / "images" / split
        self.hgt_dir = self.dataset_dir / "heights" / split
        
        if not self.img_dir.exists():
            # Fallback: dataset_dir/{split}/images/ 
            alt_img = self.dataset_dir / split / "images"
            alt_hgt = self.dataset_dir / split / "heights"
            
            if alt_img.exists():
                self.img_dir = alt_img
                self.hgt_dir = alt_hgt
            else:
                # Fallback: flat structure with no split folder
                flat_img = self.dataset_dir / "images"
                flat_hgt = self.dataset_dir / "heights"
                
                # Fallback for ISPRS Potsdam native structure
                isprs_img = self.dataset_dir / "2_Ortho_RGB"
                isprs_hgt = self.dataset_dir / "1_DSM_normalisation"
                
                if flat_img.exists():
                    self.img_dir = flat_img
                    self.hgt_dir = flat_hgt
                elif isprs_img.exists():
                    self.img_dir = isprs_img
                    self.hgt_dir = isprs_hgt
                else:
                    logger.warning("Image directory not found for split '%s' in %s", split, dataset_dir)
                    self.samples = []
                    return
        
        # Build sample pairs
        self.samples = self._build_sample_list()
        logger.info("Loaded %d paired samples from %s (split=%s, strata=%s)", 
                    len(self.samples), dataset_dir, split, strata)

    def _extract_tile_id(self, stem: str) -> str:
        """Extract a tile ID from a filename stem for matching.
        
        GAMUS naming: DC_01_25_RGB -> tile_id = DC_01_25
        ISPRS naming: top_potsdam_2_10_RGB -> tile_id = 2_10
        """
        # Strip common suffixes
        for suffix in ["_RGB", "_AGL", "_CLS", "_DSM", "_DEM"]:
            if stem.upper().endswith(suffix):
                stem = stem[:len(stem) - len(suffix)]
                break
        return stem

    def _build_sample_list(self) -> list[tuple[Path, Path]]:
        """Build a list of (image_path, height_path) pairs by matching tile IDs."""
        import re
        
        # Collect all valid image files
        img_files = sorted([
            f for f in self.img_dir.rglob("*.*") 
            if f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])
        
        # Build a lookup dict for height files: tile_id -> path
        hgt_lookup: dict[str, Path] = {}
        if self.hgt_dir.exists():
            for hgt_file in self.hgt_dir.rglob("*.*"):
                if hgt_file.is_file() and hgt_file.stat().st_size > 0 and hgt_file.suffix.lower() in self.SUPPORTED_HEIGHT_EXTS:
                    tile_id = self._extract_tile_id(hgt_file.stem)
                    hgt_lookup[tile_id] = hgt_file
        
        samples = []
        unmatched = 0
        for img_path in img_files:
            tile_id = self._extract_tile_id(img_path.stem)
            
            # Optional strata filtering
            if self.strata is not None and self.strata.lower() not in tile_id.lower():
                continue
            
            hgt_path = hgt_lookup.get(tile_id)
            if hgt_path:
                samples.append((img_path, hgt_path))
            else:
                unmatched += 1
        
        if unmatched > 0:
            logger.warning("%d image files had no matching height file.", unmatched)
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an RGB image as an H×W×3 uint8 numpy array."""
        if path.suffix.lower() == ".h5":
            arr = _load_h5_array(path)
            # H5 arrays can be (H, W, 3) or (3, H, W) — handle both
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                # Normalize float images to uint8
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            return arr
        else:
            return np.array(Image.open(path).convert("RGB"))

    def _load_height(self, path: Path) -> np.ndarray:
        """Load a height map as an H×W float32 numpy array."""
        if path.suffix.lower() == ".h5":
            arr = _load_h5_array(path)
            # Heights are typically (H, W) or (1, H, W)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            return arr.astype(np.float32)
        elif path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        else:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    return src.read(1).astype(np.float32)
            except Exception:
                img = Image.open(path).convert('L')
                return np.array(img).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, hgt_path = self.samples[idx]
        
        # Load RGB image as numpy, then PIL for transforms
        img_np = self._load_image(img_path)
        img_pil = Image.fromarray(img_np)
        
        # Load height
        hgt_np = self._load_height(hgt_path)
        hgt_tensor = torch.from_numpy(hgt_np).unsqueeze(0)  # (1, H, W)
        
        w, h = img_pil.size
        ps = self.patch_size
        if h >= ps and w >= ps:
            if self.split == "train":
                i, j = random.randint(0, h - ps), random.randint(0, w - ps)
            else:
                # Evaluation: deterministic centre crop at native GSD (resizing would change
                # the ground sample distance the head is calibrated for).
                i, j = (h - ps) // 2, (w - ps) // 2
            img_pil = TF.crop(img_pil, i, j, ps, ps)
            hgt_tensor = hgt_tensor[:, i:i + ps, j:j + ps]
        else:
            img_pil = TF.resize(img_pil, [ps, ps])
            hgt_tensor = torch.nn.functional.interpolate(
                hgt_tensor.unsqueeze(0), size=[ps, ps], mode="bilinear", align_corners=False
            ).squeeze(0)

        img_tensor = TF.to_tensor(img_pil)

        if self.split == "train" and self.augment:
            # Nadir heights are invariant to flips and 90° rotations.
            if random.random() < 0.5:
                img_tensor, hgt_tensor = img_tensor.flip(-1), hgt_tensor.flip(-1)
            if random.random() < 0.5:
                img_tensor, hgt_tensor = img_tensor.flip(-2), hgt_tensor.flip(-2)
            k = random.randint(0, 3)
            if k:
                img_tensor, hgt_tensor = torch.rot90(img_tensor, k, (-2, -1)), torch.rot90(hgt_tensor, k, (-2, -1))
            # Mild photometric jitter (sensor / illumination differences).
            img_tensor = (img_tensor * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)

        return {
            "image": img_tensor.contiguous(),
            "height": hgt_tensor.contiguous(),
            "image_path": str(img_path),
        }
