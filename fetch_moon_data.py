import rasterio
import urllib.request
from rasterio.windows import Window
from rasterio.transform import from_bounds
import numpy as np

dem_url = "/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

print("Opening DEM...")
with rasterio.open(dem_url) as src:
    print(f"DEM CRS: {src.crs}")
    print(f"DEM Bounds: {src.bounds}")
    print(f"DEM Shape: {src.width} x {src.height}")
    
    # Grab a window near the center (equator)
    w = 512
    h = 512
    cx = src.width // 2 - 4000
    cy = src.height // 2 - 2000
    
    window = Window(cx, cy, w, h)
    dem_data = src.read(1, window=window)
    transform = src.window_transform(window)
    bounds = src.window_bounds(window)
    profile = src.profile.copy()
    
    profile.update({
        'height': h,
        'width': w,
        'transform': transform,
        'tiled': True,
        'compress': 'deflate'
    })
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    
    print(f"Writing moon_dem.tif... Bounds: {bounds}")
    with rasterio.open('data/moon_surface/moon_dem.tif', 'w', **profile) as dst:
        dst.write(dem_data, 1)

print("Fetching WMS Image...")
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
