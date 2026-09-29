"""Mesh generation — the grid downsample and its handling of nodata.

Run with: pytest -q tests/test_mesher.py
"""
def test_partial_nodata_cell_keeps_its_true_height():
    """A grid cell that is only partly nodata must average its valid pixels alone.

    Filling nodata with a constant before box-averaging dragged every border cell towards
    that constant — 22 m of sag along the edge of a 60 m roof.
    """
    import numpy as np
    from heimdall.mesh.mesher import _grid

    h = np.zeros((512, 512), np.float32)
    h[:, 256:] = 60.0
    h[:51, 256:] = np.nan          # grid row 6 is 3/8 nodata and survives the mask
    g = _grid(h, 64)
    assert abs(g[6, 36] - 60.0) < 1e-3
    assert np.isnan(g[5, 36])      # wholly empty cells stay NaN for the caller to drop


def test_grid_is_unchanged_where_there_is_no_nodata():
    """The common case must take the same path as before, to the bit."""
    import numpy as np
    from PIL import Image
    from heimdall.mesh.mesher import _grid

    rng = np.random.default_rng(0)
    clean = (rng.random((900, 1100), dtype=np.float32) * 40).astype(np.float32)
    before = np.asarray(Image.fromarray(clean).resize((640, 524), Image.Resampling.BOX))
    assert np.array_equal(before, _grid(clean, 640))


def test_all_nodata_heightmap_raises_instead_of_writing_a_broken_mesh():
    import numpy as np
    import pytest
    from heimdall.mesh.mesher import build_textured_mesh

    with pytest.raises(ValueError, match="no valid pixels"):
        build_textured_mesh(np.full((128, 128), np.nan, np.float32), "unused.jpg", 0.3)
