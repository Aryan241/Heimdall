#!/usr/bin/env python3
import numpy as np
from PIL import Image

def main():
    # Create a 64x64 synthetic DEM (low-res)
    # Let's say ground is at 10m, buildings at 30m
    dem = np.ones((64, 64), dtype=np.float32) * 10.0
    
    # Add a "building"
    dem[20:40, 20:40] = 30.0
    
    # Save as float32 TIFF using PIL
    im = Image.fromarray(dem)
    im.save('data/synthetic_dem.tif')
    print("Created data/synthetic_dem.tif")

if __name__ == '__main__':
    main()
