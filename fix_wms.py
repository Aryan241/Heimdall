import rasterio
from rasterio.warp import transform_bounds
import urllib.request
import numpy as np
import os

os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

# We already have moon_dem.tif
with rasterio.open('data/moon_surface/moon_dem.tif') as src:
    w = src.width
    h = src.height
    profile = src.profile.copy()
    # Transform bounds from the DEM CRS to EPSG:4326 (lat/lon)
    bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
    print(f"Degrees Bounds: {bounds}")
    
print("Fetching WMS Image...")
# WMS BBOX for 1.1.1 is typically minx,miny,maxx,maxy (lon_min, lat_min, lon_max, lat_max)
wms_url = (
    "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map"
    "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
    "&LAYERS=LROC_WAC&STYLES=&SRS=EPSG:4326"
    f"&BBOX={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
    f"&WIDTH={w}&HEIGHT={h}&FORMAT=image/jpeg"
)
print("WMS URL:", wms_url)
req = urllib.request.Request(wms_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        img_data = resp.read()
    
    with open('data/moon_surface/moon_ortho.jpg', 'wb') as f:
        f.write(img_data)
        
    # Convert JPG to GeoTIFF using rasterio
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img_np = np.asarray(img).transpose(2, 0, 1) # (C, H, W)
    
    ortho_profile = profile.copy()
    ortho_profile.update({
        'dtype': 'uint8',
        'count': 3,
        'nodata': None
    })
    
    print("Writing moon_ortho.tif...")
    with rasterio.open('data/moon_surface/moon_ortho.tif', 'w', **ortho_profile) as dst:
        dst.write(img_np)
        
except Exception as e:
    print("WMS Error:", e)

print("Done!")
