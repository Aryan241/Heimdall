"""
Stage 8 — Mesh Generation

Converts 2D heightmaps (DSM) and RGB images into 3D meshes (.ply)
using trimesh. Generates triangular faces and applies vertex colors.
"""

import logging
import numpy as np
import trimesh

logger = logging.getLogger(__name__)

def heightmap_to_mesh(
    heightmap: np.ndarray, 
    rgb_image: np.ndarray, 
    downsample_factor: int = 2,
    z_scale: float = 1.0,
    xy_scale: float = 1.0
) -> trimesh.Trimesh:
    """
    Converts a heightmap and an RGB image into a 3D mesh.
    
    Args:
        heightmap: (H, W) array of heights (meters).
        rgb_image: (H, W, 3) uint8 array of colors.
        downsample_factor: Int > 0 to reduce mesh density (1 = native resolution, 2 = half res, etc.)
                           Native resolution creates massive .ply files that crash viewers.
        z_scale: Multiplier for the Z axis (height).
        xy_scale: Multiplier for the X and Y axes (meters per pixel).
        
    Returns:
        trimesh.Trimesh object.
    """
    logger.info("Generating 3D mesh (downsample=%d)...", downsample_factor)
    
    orig_rows, orig_cols = heightmap.shape
    
    # Downsample the arrays to save memory/disk space
    if downsample_factor > 1:
        # Use simple slicing for speed
        h_sampled = heightmap[::downsample_factor, ::downsample_factor]
        rgb_sampled = rgb_image[::downsample_factor, ::downsample_factor]
    else:
        h_sampled = heightmap
        rgb_sampled = rgb_image
        
    rows, cols = h_sampled.shape
    
    # Generate X, Y coordinates
    # We center the mesh around 0,0 for easier viewing.
    # We use original dims to ensure physical size remains constant regardless of downsampling.
    x_lin = np.linspace(-orig_cols/2, orig_cols/2, cols) * xy_scale
    y_lin = np.linspace(-orig_rows/2, orig_rows/2, rows) * xy_scale
    xx, yy = np.meshgrid(x_lin, y_lin)
    
    # Invert Y so the image isn't flipped upside down in 3D space
    yy = -yy
    
    # Flatten the arrays to create vertices
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = h_sampled.flatten() * z_scale
    
    vertices = np.column_stack((x_flat, y_flat, z_flat))
    
    # Create vertex colors
    if rgb_sampled.shape[2] == 3:
        # Add alpha channel for trimesh (Nx4)
        colors = np.column_stack((
            rgb_sampled.reshape(-1, 3), 
            np.full(len(x_flat), 255, dtype=np.uint8)
        ))
    else:
        colors = rgb_sampled.reshape(-1, 4)

    # Generate triangular faces linking the vertices
    logger.debug("Triangulating grid of %d vertices...", len(vertices))
    faces = []
    
    # Standard grid triangulation
    # For a grid of (R rows, C cols):
    # node(r, c) = r * C + c
    # Triangle 1: (r, c), (r+1, c), (r, c+1)
    # Triangle 2: (r+1, c), (r+1, c+1), (r, c+1)
    
    # Optimize by doing it vectorized
    r = np.arange(rows - 1)
    c = np.arange(cols - 1)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    
    # Vertex indices
    v00 = rr * cols + cc
    v10 = (rr + 1) * cols + cc
    v01 = rr * cols + (cc + 1)
    v11 = (rr + 1) * cols + (cc + 1)
    
    v00 = v00.flatten()
    v10 = v10.flatten()
    v01 = v01.flatten()
    v11 = v11.flatten()
    
    # Two triangles per grid cell
    tri1 = np.column_stack((v00, v10, v01))
    tri2 = np.column_stack((v10, v11, v01))
    
    faces = np.vstack((tri1, tri2))
    
    # Construct the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    logger.info("Mesh generated successfully: %d vertices, %d faces", len(vertices), len(faces))
    
    return mesh

def export_mesh(mesh: trimesh.Trimesh, filepath: str):
    """
    Exports the mesh to a file. Defaults to .ply.
    """
    logger.info("Exporting mesh to %s...", filepath)
    mesh.export(filepath)
    logger.info("✓ Saved 3D mesh: %s", filepath)
