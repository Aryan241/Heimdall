"""
Stage 8 — Mesh generation.

Builds a regular-grid terrain mesh from a height map and drapes the *full-resolution*
optical image over it as a UV texture (not per-vertex colours), exported as glTF
binary (.glb) in glTF's native frame:

    +X = image columns (east for north-up rasters)
    +Y = up (height, true metres — no vertical exaggeration is baked in)
    +Z = image rows (south for north-up rasters)

Heights are stored relative to ``height_offset`` (returned) so that vertex Y values
stay small; absolute height = vertex.y + height_offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MeshInfo:
    path: Path
    grid_shape: tuple[int, int]
    width_m: float
    depth_m: float
    height_offset: float
    vertices: int
    faces: int


def _grid(heightmap: np.ndarray, max_grid_side: int) -> np.ndarray:
    """Downsample to the mesh grid, averaging only the valid pixels of each cell.

    Returns NaN for cells with no valid pixel at all; the caller masks those out.
    """
    h, w = heightmap.shape
    s = min(1.0, max_grid_side / max(h, w))
    gh, gw = max(2, int(round(h * s))), max(2, int(round(w * s)))
    if (gh, gw) == (h, w):
        return heightmap.astype(np.float32)
    # Box-filtered downsample (BOX) keeps mean heights instead of aliasing roof edges.
    resample = Image.Resampling.BOX if s < 1 else Image.Resampling.BILINEAR
    valid = np.isfinite(heightmap)
    if valid.all():
        return np.asarray(Image.fromarray(heightmap.astype(np.float32)).resize((gw, gh), resample))
    # Filling nodata with a constant and box-averaging drags every partly-empty cell towards
    # that constant: a cell 3/8 nodata on a 60 m roof came out at 37.5 m, a 22 m sag along
    # every nodata border. Averaging height and validity separately and dividing gives the
    # mean over the valid pixels alone, which is the height that is actually known there.
    num = np.asarray(Image.fromarray(np.where(valid, heightmap, 0.0).astype(np.float32))
                     .resize((gw, gh), resample))
    den = np.asarray(Image.fromarray(valid.astype(np.float32)).resize((gw, gh), resample))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)


def build_textured_mesh(
    heightmap: np.ndarray,
    texture_path: str | Path,
    gsd_m: float,
    max_grid_side: int = 640,
    height_offset: float | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Grid mesh (≤ max_grid_side vertices per side) with UVs onto *texture_path*.

    NaN cells in *heightmap* are treated as nodata: their triangles are removed.
    """
    rows_px, cols_px = heightmap.shape
    grid = _grid(heightmap, max_grid_side)
    gr, gc = grid.shape
    nodata_grid = None
    if not np.isfinite(heightmap).all():
        m = Image.fromarray((~np.isfinite(heightmap)).astype(np.uint8) * 255)
        nodata_grid = np.asarray(m.resize((gc, gr), Image.Resampling.BOX)) > 127
    width_m, depth_m = cols_px * gsd_m, rows_px * gsd_m
    if not np.isfinite(heightmap).any():
        raise ValueError("heightmap has no valid pixels — nothing to mesh")
    if height_offset is None:
        height_offset = float(np.nanpercentile(heightmap, 1))
    if nodata_grid is not None:
        grid = np.where(nodata_grid, height_offset, grid)
    # A cell can be empty without being flagged if it rounds under the mask threshold.
    grid = np.where(np.isfinite(grid), grid, height_offset)

    u = np.linspace(0.0, 1.0, gc, dtype=np.float32)
    v = np.linspace(0.0, 1.0, gr, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x = (uu - 0.5) * width_m
    z = (vv - 0.5) * depth_m
    y = grid - height_offset
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)
    # trimesh uses OpenGL UV convention (v=0 at the bottom) and flips on glTF export.
    uv = np.column_stack([uu.ravel(), 1.0 - vv.ravel()]).astype(np.float32)

    r, c = np.meshgrid(np.arange(gr - 1), np.arange(gc - 1), indexing="ij")
    v00 = (r * gc + c).ravel()
    v10 = ((r + 1) * gc + c).ravel()
    v01 = (r * gc + c + 1).ravel()
    v11 = ((r + 1) * gc + c + 1).ravel()
    faces = np.vstack([np.column_stack([v00, v10, v01]), np.column_stack([v10, v11, v01])]).astype(np.int64)
    if nodata_grid is not None and nodata_grid.any():
        # Drop triangles touching nodata so image borders don't render as flat skirts.
        bad = nodata_grid.ravel()
        faces = faces[~(bad[faces[:, 0]] | bad[faces[:, 1]] | bad[faces[:, 2]])]
        if len(faces) == 0:
            raise ValueError("every mesh face touches nodata — nothing left to render")

    tex = Image.open(texture_path)  # opened from JPEG → trimesh embeds it without re-encoding
    material = trimesh.visual.material.PBRMaterial(
        name="optical", baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    info = {"grid_shape": (gr, gc), "width_m": width_m, "depth_m": depth_m, "height_offset": height_offset}
    logger.info("Mesh: %dx%d grid, %d vertices, %d faces, %.0f × %.0f m",
                gr, gc, len(vertices), len(faces), width_m, depth_m)
    return mesh, info


def export_glb(mesh: trimesh.Trimesh, path: str | Path, extras: dict | None = None) -> Path:
    path = Path(path).with_suffix(".glb")
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="terrain", geom_name="terrain")
    if extras:
        scene.metadata.update(extras)
    scene.export(str(path), file_type="glb")
    logger.info("GLB written: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
