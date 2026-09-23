"""
End-to-end robustness tests over awkward but realistic inputs.

These run the *whole* pipeline (backbone-only mode, so they need the Depth Anything V2
weights in the HF cache but no trained head) on small synthetic rasters: panchromatic,
16-bit multispectral with nodata, geographic CRS, rotated and south-up geotransforms,
grayscale/paletted PNGs and tiny images.

Run with: pytest -q tests/test_edge_cases.py
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_origin

from heimdall.pipeline import PipelineConfig, run

pytestmark = pytest.mark.slow

UTM = "EPSG:32643"


def _scene(h=256, w=256, seed=0) -> np.ndarray:
    """Synthetic aerial-looking scene: ground, a bright building, a dark road."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    img[:, w // 2 - 6:w // 2 + 6] = 60
    img[h // 4:h // 2, w // 4:w // 2] = (200, 190, 180)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def _write(path, data, crs=UTM, transform=from_origin(500000, 3200000, 0.5, 0.5), nodata=None, dtype=None):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=dtype or data.dtype, crs=crs, transform=transform,
                       nodata=nodata) as d:
        d.write(data)
    return path


def _run(tmp_path, src, **kw):
    cfg = PipelineConfig(input=src, output_dir=tmp_path / "out", weights=None, auto_dem=False,
                         export_mesh=True, events=False, mesh_grid=96, **kw)
    res = run(cfg)
    meta = json.loads((tmp_path / "out" / res.files["meta"]).read_text())
    assert (tmp_path / "out" / res.files["dsm"]).exists()
    assert (tmp_path / "out" / res.files["mesh"]).exists()
    assert np.isfinite(meta["stats"]["surface"]["mean"])
    return res, meta


def test_panchromatic_single_band(tmp_path):
    pan = (_scene()[..., 0]).astype(np.uint8)
    res, meta = _run(tmp_path, _write(tmp_path / "pan.tif", pan))
    assert meta["kind"] == "georeferenced"


def test_16bit_four_band_with_nodata(tmp_path):
    rgb = _scene().astype(np.uint16) * 16
    bgrn = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0], rgb[..., 0] // 2])  # B,G,R,NIR
    bgrn[:, :20, :] = 0                                                         # nodata border
    src = _write(tmp_path / "ms.tif", bgrn, nodata=0, dtype="uint16")
    res, meta = _run(tmp_path, src, band_order=[3, 2, 1])
    with rasterio.open(tmp_path / "out" / res.files["dsm"]) as d:
        arr = d.read(1)
        assert (arr[:20] == d.nodata).all()      # nodata propagated, not invented


def test_geographic_crs_degrees(tmp_path):
    deg = 1 / 3600 / 3                            # ~10 m
    src = _write(tmp_path / "wgs.tif", _scene().transpose(2, 0, 1), crs="EPSG:4326",
                 transform=from_origin(77.0, 28.6, deg, deg))
    res, meta = _run(tmp_path, src)
    assert 5 < meta["gsd"]["native_m"] < 15       # degrees converted to metres at this latitude


def test_rotated_geotransform(tmp_path):
    rot = Affine(500000, 3200000, 0, 0, 0, 0)     # placeholder, replaced below
    t = from_origin(500000, 3200000, 0.5, 0.5) * Affine.rotation(30)
    src = _write(tmp_path / "rot.tif", _scene().transpose(2, 0, 1), transform=t)
    res, meta = _run(tmp_path, src)
    assert any("rotat" in w.lower() for w in meta["warnings"]), "rotated grid should be flagged"


def test_south_up_raster(tmp_path):
    t = Affine(0.5, 0, 500000, 0, 0.5, 3200000)   # positive y step: south-up
    src = _write(tmp_path / "southup.tif", _scene().transpose(2, 0, 1), transform=t)
    _run(tmp_path, src)


def test_plain_grayscale_png(tmp_path):
    p = tmp_path / "gray.png"
    Image.fromarray(_scene()[..., 0]).convert("L").save(p)
    res, meta = _run(tmp_path, p, gsd=0.5)
    assert meta["kind"] == "plain" and meta["gsd"]["source"] == "user"


def test_paletted_png(tmp_path):
    p = tmp_path / "pal.png"
    Image.fromarray(_scene()).convert("P", palette=Image.Palette.ADAPTIVE).save(p)
    _run(tmp_path, p, gsd=0.5)


def test_tiny_image(tmp_path):
    p = tmp_path / "tiny.png"
    Image.fromarray(_scene(48, 64)).save(p)
    _run(tmp_path, p, gsd=0.5)


def test_constant_image_does_not_crash(tmp_path):
    p = tmp_path / "flat.png"
    Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(p)
    _run(tmp_path, p, gsd=0.5)


def test_max_side_caps_work_grid(tmp_path):
    src = _write(tmp_path / "big.tif", _scene(600, 600).transpose(2, 0, 1))
    res, meta = _run(tmp_path, src, max_side=128)
    assert max(meta["working_shape"]) <= 128
