import urllib.request, re
url = "https://astrogeology.usgs.gov/search/map/Moon/Apollo/Apollo15/Apollo_15_Metric_Camera_Orthomosaic_and_DEM"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    links = set(re.findall(r'href=["\'](https://astropedia\.astrogeology\.usgs\.gov/download/[^"\']+?\.tif)["\']', html))
    print("Found direct GeoTIFF links:")
    for link in links:
        print(link)
except Exception as e:
    print("Error:", e)
