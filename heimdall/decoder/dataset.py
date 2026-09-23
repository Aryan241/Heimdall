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
        backbone_cache: str | None = None,
        cache_root: str | Path | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.patch_size = patch_size
        self.split = split
        self.strata = strata
        self.augment = augment
        # Optional pre-computed frozen-backbone outputs (scripts/cache_backbone.py)
        # Cache lives beside the dataset unless cache_root is given (read-only mounts, e.g. /kaggle/input).
        root = Path(cache_root) / Path(dataset_dir).name if cache_root else Path(dataset_dir)
        self.cache_dir = (root / "backbone" / backbone_cache / split) if backbone_cache else None
        self.rng = random.Random(0)
        
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

    def _cached_depth(self, img_path: Path, img_pil: "Image.Image"):
        """Full-tile frozen-backbone output for this tile, plus the matching RGB.

        Returns (depth HxW float32, rgb PIL) or (None, img_pil). With probability 0.5 during
        training the *degraded* (satellite-like) variant is used, together with the matching
        blurred RGB, so the head also learns coarse-sensor imagery.
        """
        if self.cache_dir is None:
            return None, img_pil
        cache = self.cache_dir / f"{img_path.stem}.npz"
        if not cache.exists():
            return None, img_pil
        z = np.load(cache)
        use_deg = self.split == "train" and self.augment and random.random() < 0.5
        depth = z["degraded" if use_deg else "sharp"].astype(np.float32)
        if use_deg:
            f = float(z["factor"])
            w, h = img_pil.size
            small = img_pil.resize((max(8, int(w / f)), max(8, int(h / f))), Image.Resampling.BOX)
            img_pil = small.resize((w, h), Image.Resampling.BILINEAR)
        if depth.shape != (img_pil.size[1], img_pil.size[0]):
            depth = np.asarray(Image.fromarray(depth).resize(img_pil.size, Image.Resampling.BILINEAR))
        return depth, img_pil

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, hgt_path = self.samples[idx]
        img_pil = Image.fromarray(self._load_image(img_path))
        hgt_tensor = torch.from_numpy(self._load_height(hgt_path)).unsqueeze(0)  # (1, H, W)
        depth_np, img_pil = self._cached_depth(img_path, img_pil)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) if depth_np is not None else None

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
            if depth_tensor is not None:
                depth_tensor = depth_tensor[:, i:i + ps, j:j + ps]
        else:
            img_pil = TF.resize(img_pil, [ps, ps])
            resize = lambda t: torch.nn.functional.interpolate(
                t.unsqueeze(0), size=[ps, ps], mode="bilinear", align_corners=False).squeeze(0)
            hgt_tensor = resize(hgt_tensor)
            if depth_tensor is not None:
                depth_tensor = resize(depth_tensor)

        img_tensor = TF.to_tensor(img_pil)

        if self.split == "train" and self.augment:
            # Nadir heights are invariant to flips and 90° rotations; the frozen backbone's
            # output is transformed the same way as the image it was computed from.
            tensors = [img_tensor, hgt_tensor] + ([depth_tensor] if depth_tensor is not None else [])
            if random.random() < 0.5:
                tensors = [t.flip(-1) for t in tensors]
            if random.random() < 0.5:
                tensors = [t.flip(-2) for t in tensors]
            k = random.randint(0, 3)
            if k:
                tensors = [torch.rot90(t, k, (-2, -1)) for t in tensors]
            img_tensor, hgt_tensor = tensors[0], tensors[1]
            if depth_tensor is not None:
                depth_tensor = tensors[2]
            # Mild photometric jitter (sensor / illumination differences).
            img_tensor = (img_tensor * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)

        out = {
            "image": img_tensor.contiguous(),
            "height": hgt_tensor.contiguous(),
            "image_path": str(img_path),
        }
        if depth_tensor is not None:
            out["depth"] = depth_tensor.contiguous()
        return out
