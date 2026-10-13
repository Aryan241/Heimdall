import urllib.request
import xml.etree.ElementTree as ET

url = "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map&SERVICE=WMS&REQUEST=GetCapabilities"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    xml_data = resp.read()

root = ET.fromstring(xml_data)
ns = {"wms": "http://www.opengis.net/wms"}

for layer in root.findall(".//wms:Layer/wms:Layer", ns):
    name = layer.find("wms:Name", ns)
    title = layer.find("wms:Title", ns)
    if name is not None:
        print(f"Layer Name: {name.text}")
