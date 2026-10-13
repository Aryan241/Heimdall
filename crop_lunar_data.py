import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import numpy as np

dem_url = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
ortho_url = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC_WAC_Mosaic_global_100m_June2013.tif"

print("Opening Ortho...")
with rasterio.open(ortho_url) as src_ortho:
    # Get a 2048x2048 window somewhere interesting (e.g. Copernicus crater area roughly)
    # The image is 109164 x 327492
    width, height = src_ortho.width, src_ortho.height
    print(f"Ortho size: {width} x {height}")
    
    # Copernicus crater is at 9.62 N, 339.92 E
    # If 0,0 is top left (90N, -180E). WAC mosaic is global.
    # Let's just grab a center-ish tile to be safe
    cx, cy = int(width * 0.45), int(height * 0.45)
    window = Window(cx, cy, 2048, 2048)
    
    ortho_transform = src_ortho.window_transform(window)
    ortho_profile = src_ortho.profile.copy()
    ortho_profile.update({
        'height': 2048,
        'width': 2048,
        'transform': ortho_transform,
        'tiled': True
    })
    
    print("Reading ortho window...")
    ortho_data = src_ortho.read(1, window=window)
    bounds = src_ortho.window_bounds(window)
    
    print("Writing ortho crop...")
    with rasterio.open('data/moon_surface/lunar_ortho.tif', 'w', **ortho_profile) as dst:
        dst.write(ortho_data, 1)

print("Opening DEM...")
with rasterio.open(dem_url) as src_dem:
    print("DEM profile:", src_dem.profile)
    dem_profile = src_dem.profile.copy()
    dem_profile.update({
        'height': 2048,
        'width': 2048,
        'transform': ortho_transform,
        'tiled': True
    })
    dem_data = np.zeros((2048, 2048), dtype=dem_profile['dtype'])
    
    print("Reprojecting DEM into Ortho bounds...")
    reproject(
        source=rasterio.band(src_dem, 1),
        destination=dem_data,
        src_transform=src_dem.transform,
        src_crs=src_dem.crs,
        dst_transform=ortho_transform,
        dst_crs=ortho_profile['crs'],
        resampling=Resampling.bilinear
    )
    
    print("Writing DEM crop...")
    with rasterio.open('data/moon_surface/lunar_dem.tif', 'w', **dem_profile) as dst:
        dst.write(dem_data, 1)

print("Done! Crops saved to data/moon_surface/")
