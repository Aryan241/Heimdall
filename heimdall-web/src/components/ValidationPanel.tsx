"use client";

import { useRef, useState } from 'react';
import { fmt } from '@/lib/terrain';
import type { Metrics, Scene, ValidationReport } from '@/lib/types';

function Scatter({ x, y, label }: { x: number[]; y: number[]; label: string }) {
  const W = 300, H = 230, L = 40, B = 28, T = 8, R = 8;
  const all = [...x, ...y].filter(Number.isFinite).sort((a, b) => a - b);
  if (all.length < 2) return null;
  const lo = all[Math.floor(all.length * 0.005)];
  const hi = all[Math.floor(all.length * 0.995)];
  const span = hi - lo || 1;
  const sx = (v: number) => L + ((v - lo) / span) * (W - L - R);
  const sy = (v: number) => H - B - ((v - lo) / span) * (H - B - T);
  const ticks = [lo, lo + span / 2, hi];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Scatter plot ${label}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={T} y2={H - B} stroke="rgba(255,255,255,0.07)" />
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.07)" />
          <text x={sx(t)} y={H - B + 12} fontSize="8.5" fill="#9aa3b2" textAnchor="middle">{t.toFixed(1)}</text>
          <text x={L - 4} y={sy(t) + 3} fontSize="8.5" fill="#9aa3b2" textAnchor="end">{t.toFixed(1)}</text>
        </g>
      ))}
      <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="#fbbf24" strokeDasharray="4 3" strokeWidth="1" />
      {x.map((v, i) =>
        Number.isFinite(v) && Number.isFinite(y[i]) && v >= lo && v <= hi && y[i] >= lo && y[i] <= hi ? (
          <circle key={i} cx={sx(v)} cy={sy(y[i])} r="1.1" fill="#22d3ee" fillOpacity="0.35" />
        ) : null,
      )}
      <text x={(W + L) / 2} y={H - 3} fontSize="9" fill="#cbd5e1" textAnchor="middle">Reference ({label})</text>
      <text x={10} y={H / 2} fontSize="9" fill="#cbd5e1" textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>Heimdall</text>
    </svg>
  );
}

function Histogram({ counts, edges }: { counts: number[]; edges: number[] }) {
  const W = 300, H = 90, B = 16;
  const max = Math.max(...counts, 1);
  const bw = W / counts.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Error histogram">
      {counts.map((c, i) => (
        <rect key={i} x={i * bw + 0.5} width={bw - 1} y={H - B - (c / max) * (H - B - 4)} height={(c / max) * (H - B - 4)}
          fill={(edges[i] + edges[i + 1]) / 2 < 0 ? '#60a5fa' : '#f87171'} fillOpacity="0.8" />
      ))}
      <text x={0} y={H - 3} fontSize="8.5" fill="#9aa3b2">{edges[0].toFixed(1)} m</text>
      <text x={W / 2} y={H - 3} fontSize="8.5" fill="#9aa3b2" textAnchor="middle">0</text>
      <text x={W} y={H - 3} fontSize="8.5" fill="#9aa3b2" textAnchor="end">+{edges[edges.length - 1].toFixed(1)} m</text>
    </svg>
  );
}

function Row({ name, m }: { name: string; m?: Metrics }) {
  return (
    <tr>
      <td>{name}</td>
      <td>{fmt(m?.rmse)}</td>
      <td>{fmt(m?.mae)}</td>
      <td>{fmt(m?.bias)}</td>
      <td>{fmt(m?.nmad)}</td>
      <td>{fmt(m?.pearson_r, 3)}</td>
    </tr>
  );
}

export default function ValidationPanel({ scene, canValidate }: { scene: Scene; canValidate: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [view, setView] = useState<'surface' | 'agl'>(scene.meta.product === 'absolute_dsm' ? 'surface' : 'agl');

  const run = async (file: File) => {
    setBusy(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('reference', file);
      const res = await fetch(`/api/jobs/${scene.id}/validate`, { method: 'POST', body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || 'Validation failed');
      setReport(body.report);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <h3 className="panel-heading">Validate against reference</h3>
      <p className="muted">
        Upload a reference DSM / LiDAR raster (GeoTIFF). It is reprojected onto the prediction grid; metrics are reported
        raw, after removing the vertical-datum offset, and for above-ground heights.
      </p>
      <input ref={input} type="file" accept=".tif,.tiff" hidden
        onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
      <button className="btn-secondary" disabled={!canValidate || busy} onClick={() => input.current?.click()}>
        {busy ? 'Validating…' : 'Upload reference DSM'}
      </button>
      {!canValidate && <p className="muted small">Process an image first — the bundled demo scene is read-only.</p>}
      {error && <div className="error-box">{error}</div>}

      {report && (
        <div className="validation">
          <table className="metrics-table">
            <thead>
              <tr><th /><th>RMSE</th><th>MAE</th><th>Bias</th><th>NMAD</th><th>r</th></tr>
            </thead>
            <tbody>
              <Row name="Surface (raw)" m={report.surface_raw} />
              <Row name="Surface (offset removed)" m={report.surface_offset_removed} />
              <Row name="Above ground" m={report.above_ground} />
            </tbody>
          </table>
          <p className="muted small">
            Vertical offset (median, pred − ref): <b>{fmt(report.vertical_offset_m, 2, 'm')}</b> · overlap{' '}
            {(report.overlap_fraction * 100).toFixed(0)}% · n = {report.surface_raw.n.toLocaleString()}
          </p>
          {report.note && <p className="muted small">{report.note}</p>}

          {report.by_class && (
            <table className="metrics-table">
              <thead><tr><th>Class</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>NMAD</th><th>r</th></tr></thead>
              <tbody>
                {Object.entries(report.by_class).map(([k, m]) => <Row key={k} name={k.replace('_', ' ')} m={m} />)}
              </tbody>
            </table>
          )}

          <div className="seg">
            <button className={`chip ${view === 'surface' ? 'chip--on' : ''}`} onClick={() => setView('surface')}>Surface</button>
            <button className={`chip ${view === 'agl' ? 'chip--on' : ''}`} onClick={() => setView('agl')}>Above ground</button>
          </div>
          {view === 'surface'
            ? <Scatter x={report.scatter.ref} y={report.scatter.pred} label="m" />
            : <Scatter x={report.scatter.ref_agl} y={report.scatter.pred_agl} label="m above ground" />}
          <h4 className="subheading">Error distribution</h4>
          <Histogram counts={report.histogram.counts} edges={report.histogram.edges} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="result-img" src={`${scene.baseUrl}${report.files.error_png}`} alt="Error map" />
          <a className="download-link" href={`${scene.baseUrl}${report.files.json}`} download>Download validation report (JSON)</a>
        </div>
      )}
    </section>
  );
}
