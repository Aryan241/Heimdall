"""Fast unit tests for the geometry / calibration / evaluation core (no model weights needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from heimdall.calibration.dem import fit_gcp_correction, load_gcps, terrain_reference
from heimdall.calibration.ransac import fit_affine_transform
from heimdall.depth.depth_anything import to_closeness
from heimdall.eval.compare import compare_dsm
from heimdall.eval.metrics import height_metrics
from heimdall.geo import gsd_from_geo, scaled_transform, working_shape
from heimdall.ingestion.loader import ingest, to_uint8
from heimdall.ingestion.tiling import stitch_tiles, tile_image

UTM = "EPSG:32643"


def _write(path, arr, crs=UTM, transform=from_origin(500000, 3200000, 1.0, 1.0), nodata=None):
    arr = np.asarray(arr)
    count = 1 if arr.ndim == 2 else arr.shape[0]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1], count=count,
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr if arr.ndim == 3 else arr[None])
    return path


def test_tiling_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    img = rng.random((700, 1100, 1)).astype(np.float32)
    tiles = tile_image(img, tile_size=256, overlap=64)
    out = stitch_tiles([(t[..., 0], m) for t, m in tiles], img.shape[:2], 256, 64)
    np.testing.assert_allclose(out, img[..., 0], atol=1e-5)


def test_small_image_is_padded_not_crashing():
    img = np.zeros((100, 60, 3), np.uint8)
    tiles = tile_image(img, tile_size=512, overlap=128)
    assert len(tiles) == 1 and tiles[0][0].shape == (512, 512, 3)


def test_da3_depth_is_flipped_to_height():
    depth = np.array([[10.0, 20.0]])
    assert to_closeness(depth, "da3-metric-l")[0, 0] > to_closeness(depth, "da3-metric-l")[0, 1]
    assert to_closeness(depth, "vit-b")[0, 1] > to_closeness(depth, "vit-b")[0, 0]


def test_16bit_multiband_geotiff_band_order_and_stretch(tmp_path):
    b = np.zeros((4, 32, 32), np.uint16)
    b[0] = 100                                          # blue: constant
    b[1] = 200                                          # green: constant
    b[2] = np.linspace(1000, 4000, 32)[None, :]         # red (band 3): ramp along x
    b[2, 0, 0] = 65000                                  # saturated outlier must not crush contrast
    b[3] = 50                                           # NIR
    p = _write(tmp_path / "ms.tif", b)
    pl = ingest(p, band_order=[3, 2, 1])
    assert pl.kind == "georeferenced" and pl.image.dtype == np.uint8
    red = pl.image[10, :, 0].astype(int)
    assert red[-1] > 200 and red[1] < 30 and (np.diff(red) >= 0).all()


def test_gsd_for_projected_and_geographic():
    from heimdall.ingestion.loader import GeoInfo
    g = GeoInfo("", 32643, (0.5, 0, 0, 0, -0.5, 0), (0, 0, 1, 1), (0.5, 0.5), False)
    assert gsd_from_geo(g, (100, 100)) == pytest.approx(0.5)
    deg = 1 / 3600
    g2 = GeoInfo(rasterio.crs.CRS.from_epsg(4326).to_wkt(), 4326, (deg, 0, 77.0, 0, -deg, 29.0),
                 (77, 28.9, 77.1, 29), (deg, deg), True)
    assert 25 < gsd_from_geo(g2, (100, 100)) < 31


def test_working_shape_resamples_towards_training_gsd():
    shape, gsd = working_shape((1000, 1000), 0.66, 0.33)
    assert shape == (2000, 2000) and gsd == pytest.approx(0.33)
    shape, gsd = working_shape((1000, 1000), 5.8, 0.33)       # capped 3× up-sampling
    assert shape == (3000, 3000) and gsd == pytest.approx(5.8 / 3)
    t = scaled_transform((1, 0, 10, 0, -1, 20), (100, 100), (50, 50))
    assert t[0] == 2 and t[4] == -2 and t[2] == 10


def test_terrain_reference_keeps_metric_values_and_reprojects(tmp_path):
    # Reference DEM in EPSG:4326 at ~30 m with elevations 520–610 m.
    deg = 1 / 3600
    dem = (np.linspace(520, 610, 120)[None, :] + np.zeros((120, 1))).astype(np.float32)
    p = _write(tmp_path / "dem.tif", dem, crs="EPSG:4326", transform=from_origin(75.99, 29.01, deg, deg))
    # Image grid in UTM 43N somewhere inside that DEM.
    from rasterio.warp import transform as wt
    xs, ys = wt("EPSG:4326", UTM, [76.0], [29.0])
    img_t = (0.5, 0, xs[0], 0, -0.5, ys[0])
    terr = terrain_reference(img_t, rasterio.crs.CRS.from_string(UTM).to_wkt(), (400, 400), 0.5,
                             dem_path=str(p), opening_cells=1)
    assert terr is not None
    assert 515 < terr.dtm.min() < terr.dtm.max() < 615          # metres survive (old bug: 255)


def test_gcp_plane_fit(tmp_path):
    h, w = 100, 100
    yy, xx = np.mgrid[0:h, 0:w]
    truth = 100 + 0.05 * xx + 0.02 * yy
    surface = np.zeros((h, w), np.float32)
    csv = tmp_path / "g.csv"
    pts = [(10, 10), (10, 90), (90, 10), (90, 90), (50, 50)]
    csv.write_text("row,col,z\n" + "\n".join(f"{r},{c},{truth[r, c]}" for r, c in pts))
    corr, rep = fit_gcp_correction(surface, load_gcps(str(csv), None, None))
    assert rep["model"] == "plane" and rep["rmse_after_m"] < 1e-3
    np.testing.assert_allclose(corr, truth, atol=1e-3)


def test_ransac_recovers_affine():
    rng = np.random.default_rng(0)
    rel = rng.random((200, 200)).astype(np.float32)
    ref = 3.0 * rel + 250 + rng.normal(0, 0.05, rel.shape)
    ref[:20] += 40  # outliers (buildings)
    fit = fit_affine_transform(rel, ref)
    assert fit["scale"] == pytest.approx(3.0, abs=0.1) and fit["shift"] == pytest.approx(250, abs=0.1)


def test_metrics_include_ground_pixels():
    ref = np.zeros((50, 50)); ref[:10] = 10
    pred = np.full_like(ref, 1.0); pred[:10] = 10
    m = height_metrics(pred, ref)
    assert m["n"] == 2500 and m["mae"] == pytest.approx(0.8)   # ground error counts


def test_compare_detects_datum_offset(tmp_path):
    rng = np.random.default_rng(0)
    surf = (300 + rng.random((120, 120)) * 10).astype(np.float32)
    pred = _write(tmp_path / "pred.tif", surf)
    ref = _write(tmp_path / "ref.tif", surf - 25)
    rep = compare_dsm(pred, ref, tmp_path, "v")
    assert rep["vertical_offset_m"] == pytest.approx(25, abs=0.01)
    assert rep["surface_offset_removed"]["rmse"] < 0.01
    assert json.loads((tmp_path / "v.json").read_text())["files"]["error_png"] == "v_error.png"


def test_textured_mesh_export(tmp_path):
    from PIL import Image
    from heimdall.mesh.mesher import build_textured_mesh, export_glb
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "t.jpg")
    hm = np.zeros((64, 64), np.float32); hm[20:40, 20:40] = 12.0
    mesh, info = build_textured_mesh(hm, tmp_path / "t.jpg", gsd_m=0.5, height_offset=0.0)
    assert mesh.vertices[:, 1].max() == pytest.approx(12.0)          # true metres on +Y
    assert info["width_m"] == pytest.approx(32.0)
    assert (mesh.face_normals[:, 1] >= 0).all()                     # faces point up
    assert export_glb(mesh, tmp_path / "m.glb").stat().st_size > 1000


def test_refinement_flattens_noisy_roof_and_sharpens_walls():
    from heimdall.refine import refine_ndsm
    rng = np.random.default_rng(0)
    h, w = 120, 120
    rgb = np.full((h, w, 3), 120, np.uint8)          # grey ground
    rgb[30:90, 30:90] = (180, 60, 50)                # red roof
    truth = np.zeros((h, w), np.float32)
    truth[30:90, 30:90] = 6.0
    from scipy import ndimage
    pred = ndimage.gaussian_filter(truth, 3) + rng.normal(0, 0.6, truth.shape).astype(np.float32)  # ramps + bumps
    out, rep = refine_ndsm(pred, rgb, gsd=0.33)
    roof = out[35:85, 35:85]
    assert roof.std() < 0.05                          # flat
    assert abs(float(np.median(roof)) - 6.0) < 0.5    # at the right height
    assert out[60, 29] < 1.0 and out[60, 31] > 5.0    # vertical wall at the true edge
    assert np.abs(out - truth).mean() < np.abs(pred - truth).mean()


def test_unreadable_input_raises_friendly_error(tmp_path):
    from heimdall.ingestion.loader import UnreadableImageError
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not an image")
    with pytest.raises(UnreadableImageError, match="could not be read as an image"):
        ingest(bad)
