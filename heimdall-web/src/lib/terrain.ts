import type { SceneMeta } from '@/lib/types';

export const NODATA = -9999;

export interface LayerStats {
  min: number;
  max: number;
  p02: number;
  p98: number;
}

/**
 * Height layers sampled on the mesh grid, in vertex order (row-major), exactly as
 * written by the pipeline (`<name>_grid.bin`: float32 little-endian, one block per layer).
 */
export interface TerrainGrid {
  rows: number;
  cols: number;
  widthM: number;
  depthM: number;
  heightOffset: number;
  zScale: number;
  layers: Record<string, Float32Array>;
  stats: Record<string, LayerStats>;
}

function layerStats(a: Float32Array): LayerStats {
  const vals: number[] = [];
  const step = Math.max(1, Math.floor(a.length / 50000));
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (v === NODATA || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    if (i % step === 0) vals.push(v);
  }
  vals.sort((x, y) => x - y);
  const q = (p: number) => (vals.length ? vals[Math.min(vals.length - 1, Math.floor(p * vals.length))] : 0);
  return { min, max, p02: q(0.02), p98: q(0.98) };
}

export async function loadGrid(url: string, meta: SceneMeta): Promise<TerrainGrid> {
  const mesh = meta.mesh!;
  const [rows, cols] = mesh.grid_shape;
  const buf = await (await fetch(url)).arrayBuffer();
  const n = rows * cols;
  const layers: Record<string, Float32Array> = {};
  const stats: Record<string, LayerStats> = {};
  mesh.grid_layers.forEach((name, i) => {
    layers[name] = new Float32Array(buf, i * n * 4, n);
    stats[name] = layerStats(layers[name]);
  });
  return {
    rows, cols, widthM: mesh.width_m, depthM: mesh.depth_m,
    heightOffset: mesh.height_offset, zScale: mesh.z_scale ?? 1, layers, stats,
  };
}

/** Local mesh coordinates (x east, z south; metres) → fractional grid row/col. */
export function localToGrid(g: TerrainGrid, x: number, z: number): [number, number] {
  return [(z / g.depthM + 0.5) * (g.rows - 1), (x / g.widthM + 0.5) * (g.cols - 1)];
}

/** Bilinear sample of a layer; NaN outside the grid or on nodata. */
export function sample(g: TerrainGrid, layer: string, row: number, col: number): number {
  const a = g.layers[layer];
  if (!a || row < 0 || col < 0 || row > g.rows - 1 || col > g.cols - 1) return NaN;
  const r0 = Math.floor(row), c0 = Math.floor(col);
  const r1 = Math.min(r0 + 1, g.rows - 1), c1 = Math.min(c0 + 1, g.cols - 1);
  const fr = row - r0, fc = col - c0;
  const v = (r: number, c: number) => {
    const x = a[r * g.cols + c];
    return x === NODATA ? NaN : x;
  };
  return (v(r0, c0) * (1 - fc) + v(r0, c1) * fc) * (1 - fr) + (v(r1, c0) * (1 - fc) + v(r1, c1) * fc) * fr;
}

/** Surface value → local mesh Y (before vertical exaggeration). */
export function valueToY(g: TerrainGrid, value: number): number {
  return value * g.zScale - g.heightOffset;
}

export function sampleAt(g: TerrainGrid, layer: string, x: number, z: number): number {
  const [r, c] = localToGrid(g, x, z);
  return sample(g, layer, r, c);
}

export interface ProfilePoint {
  d: number;
  h: number;
  agl: number;
}

export function profile(g: TerrainGrid, a: [number, number], b: [number, number], n = 240): ProfilePoint[] {
  const out: ProfilePoint[] = [];
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const x = a[0] + (b[0] - a[0]) * t;
    const z = a[1] + (b[1] - a[1]) * t;
    out.push({ d: len * t, h: sampleAt(g, 'surface', x, z), agl: sampleAt(g, 'ndsm', x, z) });
  }
  return out;
}

/** Fraction and area of the scene with surface below `level`. */
export function floodStats(g: TerrainGrid, level: number) {
  const s = g.layers.surface;
  let below = 0, valid = 0;
  for (let i = 0; i < s.length; i++) {
    const v = s[i];
    if (v === NODATA || !Number.isFinite(v)) continue;
    valid++;
    if (v < level) below++;
  }
  const frac = valid ? below / valid : 0;
  return { fraction: frac, areaM2: frac * g.widthM * g.depthM * (valid / s.length) };
}

// ── Colour maps ──────────────────────────────────────────────────────────────

/** Google "Turbo" polynomial approximation. */
export function turbo(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b))];
}

const SLOPE_STOPS: [number, [number, number, number]][] = [
  [0, [0.12, 0.62, 0.33]],
  [10, [0.55, 0.8, 0.3]],
  [20, [0.98, 0.85, 0.25]],
  [35, [0.95, 0.45, 0.15]],
  [50, [0.8, 0.1, 0.12]],
];

export function slopeColor(deg: number): [number, number, number] {
  for (let i = 1; i < SLOPE_STOPS.length; i++) {
    const [d1, c1] = SLOPE_STOPS[i];
    const [d0, c0] = SLOPE_STOPS[i - 1];
    if (deg <= d1) {
      const t = (deg - d0) / (d1 - d0);
      return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
    }
  }
  return SLOPE_STOPS[SLOPE_STOPS.length - 1][1];
}

export function cssGradient(kind: 'turbo' | 'slope'): string {
  if (kind === 'slope') {
    return `linear-gradient(90deg, ${SLOPE_STOPS.map(([d, c]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(d / 50) * 100}%`).join(', ')})`;
  }
  const stops = Array.from({ length: 11 }, (_, i) => {
    const c = turbo(i / 10).map((v) => Math.round(v * 255));
    return `rgb(${c.join(',')}) ${i * 10}%`;
  });
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

export function fmt(v: number | undefined | null, digits = 2, unit = ''): string {
  if (v === undefined || v === null || !Number.isFinite(v)) return '—';
  return `${v.toFixed(digits)}${unit ? ` ${unit}` : ''}`;
}

/** Approximate lat/lon of a local point (small-area tangent-plane approximation). */
export function localToLatLon(meta: SceneMeta, g: TerrainGrid, x: number, z: number): [number, number] | null {
  if (!meta.center_latlon || !meta.working_transform) return null;
  const [lat0, lon0] = meta.center_latlon;
  // Local axes follow the raster grid; for north-up rasters +x = east, +z = south.
  const lat = lat0 - z / 110540;
  const lon = lon0 + x / (111320 * Math.cos((lat0 * Math.PI) / 180));
  return [lat, lon];
}

/** Map coordinates (raster CRS) of a local point via the working geotransform. */
export function localToCrs(meta: SceneMeta, g: TerrainGrid, x: number, z: number): [number, number] | null {
  const t = meta.working_transform;
  if (!t) return null;
  const [wh, ww] = meta.working_shape;
  const row = (z / g.depthM + 0.5) * wh;
  const col = (x / g.widthM + 0.5) * ww;
  return [t[0] * col + t[1] * row + t[2], t[3] * col + t[4] * row + t[5]];
}
