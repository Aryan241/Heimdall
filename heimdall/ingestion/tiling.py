"""
Stage 1 — Tiling: split large images into fixed-size patches with overlap,
and stitch results back into the original extent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TileMeta:
    """Metadata for one tile extracted from a larger image."""
    row_start: int
    col_start: int
    row_end: int   # exclusive
    col_end: int   # exclusive
    pad_bottom: int  # pixels padded on bottom edge
    pad_right: int   # pixels padded on right edge


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[tuple[np.ndarray, TileMeta]]:
    """Split *image* (H×W×C) into tiles of *tile_size* with *overlap*.

    Edge tiles are zero-padded to maintain consistent tile dimensions.
    Returns list of (tile_array, tile_meta) pairs.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles: list[tuple[np.ndarray, TileMeta]] = []

    for r in range(0, h, stride):
        for c in range(0, w, stride):
            r_end = min(r + tile_size, h)
            c_end = min(c + tile_size, w)
            tile = image[r:r_end, c:c_end]

            pad_b = tile_size - tile.shape[0]
            pad_r = tile_size - tile.shape[1]
            if pad_b > 0 or pad_r > 0:
                tile = np.pad(
                    tile,
                    ((0, pad_b), (0, pad_r), (0, 0)),
                    mode="reflect",
                )

            meta = TileMeta(
                row_start=r, col_start=c,
                row_end=r_end, col_end=c_end,
                pad_bottom=pad_b, pad_right=pad_r,
            )
            tiles.append((tile, meta))

    logger.info(
        "Tiled %dx%d image into %d patches (%dx%d, overlap=%d)",
        h, w, len(tiles), tile_size, tile_size, overlap,
    )
    return tiles


def stitch_tiles(
    tiles: list[tuple[np.ndarray, TileMeta]],
    original_shape: tuple[int, int],
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Stitch depth-map tiles back into the original image extent.

    Uses linear blending in overlap regions. Input tiles are 2-D (H×W) depth maps.
    """
    h, w = original_shape
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for tile, meta in tiles:
        # Remove any padding
        t = tile[: meta.row_end - meta.row_start, : meta.col_end - meta.col_start]
        r0, c0 = meta.row_start, meta.col_start
        r1, c1 = meta.row_end, meta.col_end

        # Simple averaging blend — overlap pixels get contributions from
        # multiple tiles and the weight accumulator normalizes them.
        output[r0:r1, c0:c1] += t.astype(np.float32)
        weight[r0:r1, c0:c1] += 1.0

    # Avoid division by zero (shouldn't happen with correct tiling)
    weight = np.maximum(weight, 1e-8)
    return output / weight
