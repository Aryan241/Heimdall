"""
Stage 1 — Tiling: split large images into fixed-size patches with overlap,
and stitch per-tile predictions back with feathered (raised-cosine) blending.

The last tile along each axis is placed flush with the image edge, so tiles are
only padded when the whole image is smaller than one tile.
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
    pad_bottom: int
    pad_right: int


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return sorted(set(starts))


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
) -> list[tuple[np.ndarray, TileMeta]]:
    """Split *image* (H×W×C) into tile_size² tiles overlapping by *overlap* pixels."""
    h, w = image.shape[:2]
    stride = max(1, tile_size - overlap)
    tiles: list[tuple[np.ndarray, TileMeta]] = []
    for r in _starts(h, tile_size, stride):
        for c in _starts(w, tile_size, stride):
            r_end, c_end = min(r + tile_size, h), min(c + tile_size, w)
            tile = image[r:r_end, c:c_end]
            pad_b, pad_r = tile_size - tile.shape[0], tile_size - tile.shape[1]
            if pad_b > 0 or pad_r > 0:
                pad = ((0, pad_b), (0, pad_r)) + ((0, 0),) * (image.ndim - 2)
                tile = np.pad(tile, pad, mode="symmetric")
            tiles.append((tile, TileMeta(r, c, r_end, c_end, pad_b, pad_r)))
    logger.info("Tiled %dx%d image into %d patches (%d², overlap=%d)", h, w, len(tiles), tile_size, overlap)
    return tiles


def _ramp(n: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    k = min(overlap, n // 2)
    if k > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
        if not at_start:
            w[:k] = ramp
        if not at_end:
            w[n - k:] = ramp[::-1]
    return np.maximum(w, 1e-3)


def stitch_tiles(
    tiles: list[tuple[np.ndarray, TileMeta]],
    original_shape: tuple[int, int],
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    """Blend 2-D tile predictions into the full extent with raised-cosine weights."""
    h, w = original_shape
    output = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    for tile, m in tiles:
        th, tw = m.row_end - m.row_start, m.col_end - m.col_start
        t = tile[:th, :tw].astype(np.float64)
        wr = _ramp(th, overlap, m.row_start == 0, m.row_end == h)
        wc = _ramp(tw, overlap, m.col_start == 0, m.col_end == w)
        wt = np.outer(wr, wc)
        output[m.row_start:m.row_end, m.col_start:m.col_end] += t * wt
        weight[m.row_start:m.row_end, m.col_start:m.col_end] += wt
    return (output / np.maximum(weight, 1e-12)).astype(np.float32)
