import urllib.request, re

url = "http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/APOLLO15/"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        html = response.read().decode()
    links = re.findall(r'href=["\'](.*?\.TIF)["\']', html, re.IGNORECASE)
    print("Found files:", links[:10])
except Exception as e:
    print("Error:", e)
