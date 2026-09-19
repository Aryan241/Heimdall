/** Shape of `<name>_meta.json` written by heimdall/pipeline.py (version 2). */
export interface Stats {
  min: number;
  max: number;
  mean: number;
  std: number;
  p02: number;
  p50: number;
  p98: number;
}

export type Product = 'absolute_dsm' | 'ndsm' | 'relative_dsm';

export interface MeshMeta {
  grid_shape: [number, number];
  width_m: number;
  depth_m: number;
  height_offset: number;
  grid_layers: string[];
  z_scale: number;
  vertices: number;
  faces: number;
}

export interface SceneMeta {
  version: number;
  input: string;
  kind: 'georeferenced' | 'plain';
  product: Product;
  product_description: string;
  units: 'm' | 'relative';
  mode: 'head' | 'relative';
  model: string | null;
  weights: string | null;
  native_shape: [number, number];
  working_shape: [number, number];
  gsd: { source: string; native_m: number; working_m: number; detections?: number };
  crs_epsg: number | null;
  working_transform: number[] | null;
  center_latlon: [number, number] | null;
  bounds: number[] | null;
  calibration: Record<string, unknown> & {
    method?: string;
    terrain_source?: string;
    gcp?: { used: number; model: string; rmse_before_m: number; rmse_after_m: number };
  };
  stats: { surface: Stats; ndsm?: Stats; dtm?: Stats; built_fraction?: number };
  mesh: MeshMeta | null;
  files: Record<string, string>;
  warnings: string[];
  timings_s: Record<string, number>;
}

/** A loaded scene: metadata plus the URL prefix its files are served from. */
export interface Scene {
  id: string;
  baseUrl: string;
  meta: SceneMeta;
}

export interface JobSummary {
  id: string;
  input: string;
  created: string;
  status: 'running' | 'done' | 'error';
  product?: Product;
  meta?: string;
}

export interface Metrics {
  n: number;
  rmse?: number;
  mae?: number;
  bias?: number;
  median_abs?: number;
  nmad?: number;
  le90?: number;
  rmse_debiased?: number;
  pearson_r?: number;
  r2?: number;
  delta1?: number;
}

export interface ValidationReport {
  reference: string;
  prediction: string;
  overlap_fraction: number;
  gsd_m: number;
  surface_raw: Metrics;
  surface_offset_removed: Metrics;
  vertical_offset_m: number;
  above_ground: Metrics;
  by_class?: Record<string, Metrics>;
  by_class_above_ground?: Record<string, Metrics>;
  scatter: { ref: number[]; pred: number[]; ref_agl: number[]; pred_agl: number[] };
  histogram: { counts: number[]; edges: number[] };
  files: { error_png: string; json: string };
  note?: string;
}

export interface PipelineEvent {
  event: 'stage' | 'progress' | 'info' | 'warning' | 'done' | 'error' | 'validation';
  stage?: string;
  message?: string;
  current?: number;
  total?: number;
  [key: string]: unknown;
}
