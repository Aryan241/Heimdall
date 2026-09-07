"""
Stage 3 — Unified Dataset Loader for Domain Adaptation

Loads RGB imagery and ground-truth height maps (DFC2019 / ISPRS).
Handles random cropping, normalization, and strata filtering.
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

class RemoteSensingHeightDataset(Dataset):
    """Unified dataset for DFC2019 and ISPRS Potsdam/Vaihingen.
    
    Assumes a directory structure:
        dataset_dir/
            images/   (RGB .jpg/.png)
            heights/  (Grayscale or float .png/.npy/.tif)
    """
    def __init__(
        self,
        dataset_dir: str | Path,
        patch_size: int = 512,
        split: Literal["train", "val"] = "train",
        strata: str | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.patch_size = patch_size
        self.split = split
        self.strata = strata
        
        self.img_dir = self.dataset_dir / split / "images"
        self.hgt_dir = self.dataset_dir / split / "heights"
        
        if not self.img_dir.exists():
            # Fallback for flat dataset structures without a train/val split folder
            fallback_img_dir = self.dataset_dir / "images"
            fallback_hgt_dir = self.dataset_dir / "heights"
            
            # Fallback for ISPRS Potsdam native structure
            isprs_img_dir = self.dataset_dir / "2_Ortho_RGB"
            isprs_hgt_dir = self.dataset_dir / "1_DSM_normalisation"
            
            if fallback_img_dir.exists():
                self.img_dir = fallback_img_dir
                self.hgt_dir = fallback_hgt_dir
            elif isprs_img_dir.exists():
                self.img_dir = isprs_img_dir
                self.hgt_dir = isprs_hgt_dir
            else:
                logger.warning("Image directory %s not found. (Expected if running dummy test)", self.img_dir)
                self.samples = []
                return
                
        import re
        
        # Match image files with height files by stem or ISPRS tile ID
        # Use rglob to search recursively in case the dataset has nested folders (e.g. 2_Ortho_RGB/2_Ortho_RGB/)
        img_files = sorted([f for f in self.img_dir.rglob("*.*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]])
        self.samples = []
        for img_path in img_files:
            stem = img_path.stem
            hgt_path = None
            
            # Extract tile ID like '2_10' if this is an ISPRS dataset (e.g., top_potsdam_2_10_RGB)
            # Make sure it handles '02_10' as well by stripping leading zeros
            match = re.search(r'(\d+_\d+)', stem)
            tile_id = match.group(1) if match else stem
            # Strip leading zeros from parts (e.g. 02_10 -> 2_10) to ensure robust matching
            normalized_tile_id = "_".join([str(int(p)) for p in tile_id.split('_')]) if '_' in tile_id else tile_id
            
            # Search for a height file containing the tile ID or exact stem
            # Use rglob recursively here as well. Added .jpg for ISPRS ownapproach visualization DSMs.
            for hgt_file in self.hgt_dir.rglob("*.*"):
                if hgt_file.is_file() and hgt_file.suffix.lower() in [".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
                    hgt_stem_norm = "_".join([str(int(p)) for p in re.findall(r'\d+', hgt_file.stem)]) if re.findall(r'\d+', hgt_file.stem) else hgt_file.stem
                    if normalized_tile_id in hgt_stem_norm or stem == hgt_file.stem or tile_id in hgt_file.stem:
                        hgt_path = hgt_file
                        break
            
            if hgt_path:
                # Optional: Check if filename contains strata string if filtering
                if self.strata is None or self.strata.lower() in stem.lower():
                    self.samples.append((img_path, hgt_path))
                    
        logger.info("Loaded %d paired samples from %s (split=%s, strata=%s)", 
                    len(self.samples), dataset_dir, split, strata)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_height(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        try:
            # Using rasterio since heights are often single-channel float32 TIFFs
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)
        except Exception:
            # Fallback to PIL for JPEGs/PNGs if rasterio/GDAL rejects them
            from PIL import Image
            img = Image.open(path).convert('L')
            return np.array(img).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, hgt_path = self.samples[idx]
        
        # Load RGB
        img_pil = Image.open(img_path).convert("RGB")
        
        # Load height
        hgt_np = self._load_height(hgt_path)
        hgt_tensor = torch.from_numpy(hgt_np).unsqueeze(0) # (1, H, W)
        
        # Random Crop for training
        if self.split == "train":
            w, h = img_pil.size
            if h >= self.patch_size and w >= self.patch_size:
                i = random.randint(0, h - self.patch_size)
                j = random.randint(0, w - self.patch_size)
                img_pil = TF.crop(img_pil, i, j, self.patch_size, self.patch_size)
                hgt_tensor = hgt_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
            else:
                img_pil = TF.resize(img_pil, [self.patch_size, self.patch_size])
                hgt_tensor = F.interpolate(hgt_tensor.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear').squeeze(0)
        else:
            # Validation: just resize to patch_size for batching simplicity, 
            # or keep original (but batch size must be 1)
            img_pil = TF.resize(img_pil, [self.patch_size, self.patch_size])
            hgt_tensor = torch.nn.functional.interpolate(
                hgt_tensor.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear'
            ).squeeze(0)

        # Convert RGB to tensor [0, 1]
        img_tensor = TF.to_tensor(img_pil)
        
        return {
            "image": img_tensor,   # (3, H, W)
            "height": hgt_tensor,  # (1, H, W)
            "image_path": str(img_path)
        }
