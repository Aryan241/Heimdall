"use client";

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useRef, useState } from 'react';
import ValidationPanel from '@/components/ValidationPanel';
import { fmt, loadGrid, type TerrainGrid } from '@/lib/terrain';
import type { JobSummary, PipelineEvent, Scene, SceneMeta } from '@/lib/types';

const TerrainViewer = dynamic(() => import('@/components/TerrainViewer'), {
  ssr: false,
  loading: () => <div className="viewer-empty">Initialising 3D engine…</div>,
});

const STAGES: { key: string; label: string }[] = [
  { key: 'ingest', label: 'Ingest & georeferencing' },
  { key: 'gsd', label: 'Ground sample distance' },
  { key: 'depth', label: 'Height estimation (DA3 + head)' },
  { key: 'calibration', label: 'Absolute calibration' },
  { key: 'outputs', label: 'GeoTIFF / PNG export' },
  { key: 'mesh', label: 'Textured 3D mesh' },
];

const PRODUCT_LABEL: Record<string, string> = {
  absolute_dsm: 'Absolute DSM',
  ndsm: 'nDSM (height above ground)',
  relative_dsm: 'Relative DSM (rDSM)',
};

const FILE_LABEL: Record<string, string> = {
  dsm: 'Elevation model (GeoTIFF)',
  ndsm: 'Height above ground (GeoTIFF)',
  dtm: 'Terrain datum (GeoTIFF)',
  height_png: '16-bit height PNG',
  preview: 'Colour relief preview (PNG)',
  mesh: 'Textured 3D mesh (GLB)',
  texture: 'Texture (JPEG)',
  meta: 'Metadata (JSON)',
};

interface Progress {
  active: string | null;
  done: Set<string>;
  tiles: { current: number; total: number } | null;
  message: string;
}

export default function Home() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [grid, setGrid] = useState<TerrainGrid | null>(null);
  const [loadError, setLoadError] = useState('');
  const [tab, setTab] = useState<'3d' | 'maps' | 'report'>('3d');
  const [jobs, setJobs] = useState<JobSummary[]>([]);

  const [file, setFile] = useState<File | null>(null);
  const [dem, setDem] = useState<File | null>(null);
  const [gcps, setGcps] = useState<File | null>(null);
  const [opts, setOpts] = useState({ gsd: '', bands: '', tta: false, refine: true, autoDem: true, mode: 'head' as 'head' | 'relative' });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [dragging, setDragging] = useState(false);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress>({ active: null, done: new Set(), tiles: null, message: '' });
  const [warnings, setWarnings] = useState<string[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [started, setStarted] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const imgInput = useRef<HTMLInputElement>(null);
  const demInput = useRef<HTMLInputElement>(null);
  const gcpInput = useRef<HTMLInputElement>(null);

  const openScene = useCallback(async (id: string, baseUrl: string, metaFile: string) => {
    setLoadError('');
    try {
      const res = await fetch(baseUrl + metaFile);
      if (!res.ok) throw new Error(`metadata not found (${res.status})`);
      const meta = (await res.json()) as SceneMeta;
      const next: Scene = { id, baseUrl, meta };
      const g = meta.mesh && meta.files.grid ? await loadGrid(baseUrl + meta.files.grid, meta) : null;
      setScene(next);
      setGrid(g);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs');
      if (res.ok) setJobs((await res.json()).jobs ?? []);
    } catch {}
  }, []);

  useEffect(() => {
    // Initial load: bundled demo scene + job history (async, so state updates happen after mount).
    const t = setTimeout(() => {
      openScene('demo', '/demo/', 'demo_meta.json');
      refreshJobs();
    }, 0);
    return () => clearTimeout(t);
  }, [openScene, refreshJobs]);

  useEffect(() => {
    if (!running || !started) return;
    const t = setInterval(() => setElapsed((Date.now() - started) / 1000), 500);
    return () => clearInterval(t);
  }, [running, started]);

  const handleEvent = (e: PipelineEvent) => {
    if (e.event === 'stage' && e.stage) {
      setProgress((p) => {
        const done = new Set(p.done);
        if (p.active && p.active !== e.stage) done.add(p.active);
        return { active: e.stage!, done, tiles: e.stage === p.active ? p.tiles : null, message: e.message ?? '' };
      });
    } else if (e.event === 'progress' && e.total) {
      setProgress((p) => ({ ...p, tiles: { current: e.current ?? 0, total: e.total! } }));
    } else if (e.event === 'warning' && e.message) {
      setWarnings((w) => [...w, e.message!]);
    }
  };

  const process = async () => {
    if (!file) return;
    setRunning(true);
    setError('');
    setWarnings([]);
    setLogs([]);
    setProgress({ active: 'ingest', done: new Set(), tiles: null, message: 'Uploading…' });
    setStarted(Date.now());
    setElapsed(0);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const fd = new FormData();
      fd.append('image', file);
      if (dem) fd.append('reference_dem', dem);
      if (gcps) fd.append('gcps', gcps);
      fd.append('options', JSON.stringify({
        gsd: opts.gsd ? parseFloat(opts.gsd) : undefined,
        bands: opts.bands || undefined,
        tta: opts.tta,
        refine: opts.refine,
        autoDem: opts.autoDem,
        mode: opts.mode,
      }));
      const res = await fetch('/api/jobs', { method: 'POST', body: fd, signal: ctrl.signal });
      if (!res.ok || !res.body) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Server error ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let jobId = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split('\n\n');
        buffer = chunks.pop() ?? '';
        for (const chunk of chunks) {
          let type = 'message';
          let data = '';
          for (const line of chunk.split('\n')) {
            if (line.startsWith('event: ')) type = line.slice(7);
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (!data) continue;
          const payload = JSON.parse(data);
          if (type === 'job') jobId = payload.id;
          else if (type === 'event') handleEvent(payload as PipelineEvent);
          else if (type === 'log') setLogs((l) => (l.length > 400 ? [...l.slice(-300), payload.text] : [...l, payload.text]));
          else if (type === 'error') throw new Error(payload.message);
          else if (type === 'done') {
            setProgress((p) => ({ active: null, done: new Set(STAGES.map((s) => s.key)), tiles: p.tiles, message: 'Complete' }));
            await openScene(jobId, `/api/jobs/${jobId}/files/`, payload.meta);
            setTab('3d');
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      abortRef.current = null;
      refreshJobs();
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  const meta = scene?.meta;
  const s = meta?.stats.surface;
  const units = meta?.units === 'm' ? 'm' : 'rel';

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo-mark" aria-hidden>◆</span>
          <div>
            <div className="logo text-gradient">HEIMDALL</div>
            <div className="header-subtitle">Single-view height estimation &amp; 3D flythrough</div>
          </div>
        </div>
        {meta && (
          <div className="header-scene">
            <span className={`badge badge--${meta.product}`}>{PRODUCT_LABEL[meta.product]}</span>
            <span className="muted">{meta.input}{scene?.id === 'demo' ? ' · demo' : ''}</span>
          </div>
        )}
      </header>

      <main className="layout">
        <aside className="sidebar">
          {/* ── New job ── */}
          <section className="panel">
            <h3 className="panel-heading">New processing job</h3>
            <input ref={imgInput} type="file" hidden accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
              onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])} />
            <input ref={demInput} type="file" hidden accept=".tif,.tiff"
              onChange={(e) => setDem(e.target.files?.[0] ?? null)} />
            <input ref={gcpInput} type="file" hidden accept=".csv,.txt"
              onChange={(e) => setGcps(e.target.files?.[0] ?? null)} />

            <div
              className={`dropzone ${file ? 'dropzone--set' : ''} ${dragging ? 'dropzone--drag' : ''}`}
              onClick={() => !running && imgInput.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
            >
              <div className="dropzone-title">{file ? file.name : 'Drop an optical image here'}</div>
              <div className="dropzone-meta">
                {file ? `${(file.size / 1048576).toFixed(1)} MB` : 'GeoTIFF → absolute DSM · PNG/JPG → relative DSM'}
              </div>
            </div>

            <div className="aux-inputs">
              <button className="aux" onClick={() => demInput.current?.click()} disabled={running}>
                <span>Reference DEM</span>
                <small>{dem ? dem.name : 'optional · SRTM / Copernicus / CartoDEM'}</small>
              </button>
              {dem && <button className="aux-clear" onClick={() => setDem(null)} aria-label="Remove DEM">×</button>}
              <button className="aux" onClick={() => gcpInput.current?.click()} disabled={running}>
                <span>Ground control points</span>
                <small>{gcps ? gcps.name : 'optional · CSV lon,lat,z | x,y,z | row,col,z'}</small>
              </button>
              {gcps && <button className="aux-clear" onClick={() => setGcps(null)} aria-label="Remove GCPs">×</button>}
            </div>

            <button className="link-btn" onClick={() => setShowAdvanced((v) => !v)}>
              {showAdvanced ? '▾' : '▸'} Advanced options
            </button>
            {showAdvanced && (
              <div className="advanced">
                <label>GSD override (m/px)
                  <input type="number" step="0.01" min="0" placeholder="auto" value={opts.gsd}
                    onChange={(e) => setOpts({ ...opts, gsd: e.target.value })} />
                </label>
                <label>Band order (R,G,B)
                  <input type="text" placeholder="auto, e.g. 3,2,1" value={opts.bands}
                    onChange={(e) => setOpts({ ...opts, bands: e.target.value.replace(/\s/g, '') })} />
                </label>
                <label>Height model
                  <select value={opts.mode} onChange={(e) => setOpts({ ...opts, mode: e.target.value as 'head' | 'relative' })}>
                    <option value="head">Trained head (metric nDSM)</option>
                    <option value="relative">Backbone only (relative + RANSAC)</option>
                  </select>
                </label>
                <label className="check">
                  <input type="checkbox" checked={opts.autoDem} onChange={(e) => setOpts({ ...opts, autoDem: e.target.checked })} />
                  Auto-download Copernicus GLO-30 terrain
                </label>
                <label className="check">
                  <input type="checkbox" checked={opts.refine} onChange={(e) => setOpts({ ...opts, refine: e.target.checked })} />
                  Regularise roofs &amp; walls (flat planes, sharp edges)
                </label>
                <label className="check">
                  <input type="checkbox" checked={opts.tta} onChange={(e) => setOpts({ ...opts, tta: e.target.checked })} />
                  Test-time augmentation (slower, more accurate)
                </label>
              </div>
            )}

            {running ? (
              <button className="btn-primary" onClick={() => abortRef.current?.abort()}>Cancel</button>
            ) : (
              <button className="btn-primary" disabled={!file} onClick={process}>Process image</button>
            )}
            {error && <div className="error-box"><strong>Error:</strong> {error}</div>}
          </section>

          {/* ── Progress ── */}
          {(running || progress.done.size > 0 || error) && (
            <section className="panel">
              <h3 className="panel-heading">
                Pipeline {running && <span className="muted small">· {elapsed.toFixed(0)} s</span>}
              </h3>
              <ol className="stages">
                {STAGES.map((st) => {
                  const state = progress.done.has(st.key) ? 'done' : progress.active === st.key ? (error ? 'error' : 'active') : 'todo';
                  return (
                    <li key={st.key} className={`stage stage--${state}`}>
                      <span className="stage-dot" />
                      <div>
                        <div>{st.label}</div>
                        {state === 'active' && progress.message && <div className="small muted">{progress.message}</div>}
                        {st.key === 'depth' && progress.tiles && state !== 'todo' && (
                          <div className="bar"><div style={{ width: `${(100 * progress.tiles.current) / progress.tiles.total}%` }} />
                            <span>{progress.tiles.current}/{progress.tiles.total} tiles</span></div>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ol>
              {warnings.map((w, i) => <div key={i} className="warn-box">{w}</div>)}
              {logs.length > 0 && (
                <details className="logs">
                  <summary>Log ({logs.length} lines)</summary>
                  <pre>{logs.slice(-150).join('\n')}</pre>
                </details>
              )}
            </section>
          )}

          {/* ── Result summary ── */}
          {meta && scene && (
            <section className="panel">
              <h3 className="panel-heading">Result</h3>
              <p className="muted small">{meta.product_description}</p>
              <div className="stat-grid">
                <div><span>Min</span><b>{fmt(s?.min, 2)}</b><em>{units}</em></div>
                <div><span>Max</span><b>{fmt(s?.max, 2)}</b><em>{units}</em></div>
                <div><span>Mean</span><b>{fmt(s?.mean, 2)}</b><em>{units}</em></div>
                {meta.stats.ndsm && <div><span>Tallest object</span><b>{fmt(meta.stats.ndsm.p98, 1)}</b><em>m (p98)</em></div>}
                {meta.stats.built_fraction !== undefined && (
                  <div><span>Built / canopy</span><b>{(meta.stats.built_fraction * 100).toFixed(0)}</b><em>% &gt; 2.5 m</em></div>
                )}
                <div><span>Working GSD</span><b>{fmt(meta.gsd.working_m, 2)}</b><em>m/px</em></div>
              </div>
              <dl className="details">
                <dt>Input</dt><dd>{meta.kind === 'georeferenced' ? `GeoTIFF · EPSG:${meta.crs_epsg ?? '?'}` : 'Plain image'} · {meta.native_shape[1]}×{meta.native_shape[0]} px</dd>
                <dt>GSD</dt><dd>{fmt(meta.gsd.native_m, 3)} m/px native ({meta.gsd.source})</dd>
                {meta.center_latlon && <><dt>Centre</dt><dd>{meta.center_latlon[0].toFixed(5)}°, {meta.center_latlon[1].toFixed(5)}°</dd></>}
                <dt>Model</dt><dd>{meta.mode === 'head' ? `Depth Anything V3 + Heimdall head (${meta.weights})` : `${meta.model} (relative)`}</dd>
                <dt>Calibration</dt><dd>{meta.calibration.method ?? 'none'}{meta.calibration.terrain_source ? ` · ${meta.calibration.terrain_source}` : ''}</dd>
                {meta.calibration.gcp && <><dt>GCPs</dt><dd>{meta.calibration.gcp.used} used ({meta.calibration.gcp.model}) · RMSE {fmt(meta.calibration.gcp.rmse_before_m)} → {fmt(meta.calibration.gcp.rmse_after_m)} m</dd></>}
                <dt>Runtime</dt><dd>{fmt(meta.timings_s.total, 1)} s</dd>
              </dl>
              {meta.warnings.length > 0 && meta.warnings.map((w, i) => <div key={i} className="warn-box">{w}</div>)}
              <h4 className="subheading">Downloads</h4>
              <div className="downloads">
                {Object.entries(meta.files).filter(([k]) => FILE_LABEL[k]).map(([k, name]) => (
                  <a key={k} className="download-link" href={scene.baseUrl + name} download={name}>{FILE_LABEL[k]}</a>
                ))}
              </div>
            </section>
          )}

          {scene && <ValidationPanel key={scene.id} scene={scene} canValidate={scene.id !== 'demo'} />}

          {jobs.length > 0 && (
            <section className="panel">
              <h3 className="panel-heading">Recent jobs</h3>
              <ul className="jobs">
                {jobs.map((j) => (
                  <li key={j.id}>
                    <button disabled={j.status !== 'done' || !j.meta}
                      className={scene?.id === j.id ? 'job job--on' : 'job'}
                      onClick={() => j.meta && openScene(j.id, `/api/jobs/${j.id}/files/`, j.meta)}>
                      <span>{j.input}</span>
                      <small>{new Date(j.created).toLocaleString()} · {j.status}</small>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>

        <section className="stage-area">
          <div className="tabs">
            {(['3d', 'maps', 'report'] as const).map((t) => (
              <button key={t} className={`tab ${tab === t ? 'tab--on' : ''}`} onClick={() => setTab(t)}>
                {{ '3d': '3D flythrough', maps: '2D maps', report: 'Metadata' }[t]}
              </button>
            ))}
          </div>
          <div className="tab-body">
            {loadError && <div className="viewer-empty">Could not load scene: {loadError}</div>}
            {!loadError && !scene && <div className="viewer-empty">Loading…</div>}
            {scene && tab === '3d' && (grid
              ? <TerrainViewer key={scene.id} scene={scene} grid={grid} />
              : <div className="viewer-empty">This result has no mesh.</div>)}
            {scene && tab === 'maps' && (
              <div className="maps">
                {/* eslint-disable @next/next/no-img-element */}
                <figure><img src={scene.baseUrl + scene.meta.files.preview} alt="Colour relief" /><figcaption>Colour relief (hill-shaded)</figcaption></figure>
                <figure><img src={scene.baseUrl + scene.meta.files.texture} alt="Optical input" /><figcaption>Optical input</figcaption></figure>
                {/* eslint-enable @next/next/no-img-element */}
              </div>
            )}
            {scene && tab === 'report' && <pre className="report">{JSON.stringify(scene.meta, null, 2)}</pre>}
          </div>
        </section>
      </main>
    </div>
  );
}
