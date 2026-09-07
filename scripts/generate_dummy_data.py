#!/usr/bin/env python3
"""
Generate a small dummy dataset to test the training script locally.
"""
import os
import numpy as np
from PIL import Image

def generate_dummy_dataset(base_dir="data/dummy_dataset", num_samples=5):
    img_dir = os.path.join(base_dir, "train", "images")
    hgt_dir = os.path.join(base_dir, "train", "heights")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(hgt_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create a random RGB image (512x512)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img_path = os.path.join(img_dir, f"sample_urban_{i:03d}.png")
        Image.fromarray(img).save(img_path)
        
        # Create a matching random height map (float32)
        hgt = np.random.rand(512, 512).astype(np.float32) * 20.0 # 0 to 20 meters
        hgt_path = os.path.join(hgt_dir, f"sample_urban_{i:03d}.npy")
        np.save(hgt_path, hgt)
        
    print(f"Generated {num_samples} dummy samples in {base_dir}")

if __name__ == "__main__":
    generate_dummy_dataset()
