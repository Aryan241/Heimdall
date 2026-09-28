"use client";

import { useEffect, useState } from 'react';
import { fmt } from '@/lib/terrain';

interface SiteRow {
  site: string; landscape: string | null;
  rmse: number; mae: number; bias: number; pearson_r: number; predict_zero_rmse: number;
}
interface GsdRow { gsd: number; sites: number; rmse: number; mae: number; pearson_r: number; predict_zero_rmse: number }
interface Benchmarks {
  generated: string;
  model: { backbone: string; head: string; trained_on: string; val_rmse: number | null };
  lidar_sites: SiteRow[];
  by_gsd: GsdRow[];
  site_maps: Record<string, string>;
  gamus_test: {
    tiles: number;
    overall: Record<string, number>;
    predict_zero: Record<string, number>;
    by_landscape: Record<string, { rmse: number; mae: number; pearson_r: number }>;
    by_pixel_class: Record<string, { rmse: number; mae: number; bias: number }>;
  } | null;
}

const pretty = (s: string) => s.replace(/_/g, ' ');
// eval.py folds road pixels into `ground` (both are bare-earth level) and also reports road on
// its own, so the two rows overlap. Say so rather than letting the table imply they are disjoint.
const CLASS_LABEL: Record<string, string> = { ground: 'ground (incl. road)' };

export default function AccuracyPanel() {
  const [data, setData] = useState<Benchmarks | null>(null);
  const [err, setErr] = useState('');
  const [site, setSite] = useState<string | null>(null);

  useEffect(() => {
    fetch('/benchmarks.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`not found (${r.status})`))))
      .then((d: Benchmarks) => { setData(d); setSite(d.lidar_sites[0]?.site ?? null); })
      .catch((e) => setErr(e.message));
  }, []);

  if (err) return <p className="muted">Benchmark results unavailable: {err}</p>;
  if (!data) return <p className="muted">Loading measured results…</p>;

  const g = data.gamus_test;
  const zero = g?.predict_zero.rmse;

  return (
    <div className="accuracy">
      <section className="panel">
        <h3 className="panel-heading">How accurate is this model?</h3>
        <p className="muted small">
          Every number here is measured on data the model never saw in training, and compared against
          the trivial baseline of predicting zero height everywhere — so the figures mean something
          rather than just sounding small.
        </p>
        <div className="headline-metrics">
          <div><strong>{fmt(g?.overall.median_abs)} m</strong><span>half of all pixels are within this</span></div>
          <div><strong>{fmt(g?.overall.mae)} m</strong><span>average error</span></div>
          <div><strong>{fmt(g?.overall.pearson_r)}</strong><span>correlation with real height</span></div>
          <div><strong>{fmt(g?.overall.rmse)} m</strong><span>RMSE &mdash; vs {fmt(zero)} m for predicting zero</span></div>
        </div>
        <p className="muted small">
          {data.model.backbone} + {data.model.head}. Trained on {data.model.trained_on}.
          Held-out GAMUS test split, {g?.tiles} tiles.
        </p>
      </section>

      <section className="panel">
        <h3 className="panel-heading">Against airborne LiDAR, by landscape</h3>
        <p className="muted small">
          Eight Dutch sites, scored against AHN laser-scan measurements. None were used for training.
          Click a row to see where the errors are.
        </p>
        <table className="metrics-table">
          <thead>
            <tr><th>Site</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>r</th><th>Predict-0</th></tr>
          </thead>
          <tbody>
            {data.lidar_sites.map((s) => (
              <tr key={s.site} onClick={() => setSite(s.site)}
                className={site === s.site ? 'row--on' : undefined} style={{ cursor: 'pointer' }}>
                <td>{pretty(s.site)}<br /><small className="muted">{s.landscape}</small></td>
                <td>{fmt(s.rmse)} m</td>
                <td>{fmt(s.mae)} m</td>
                <td>{fmt(s.bias)} m</td>
                <td>{fmt(s.pearson_r)}</td>
                <td className="muted">{fmt(s.predict_zero_rmse)} m</td>
              </tr>
            ))}
          </tbody>
        </table>
        {site && data.site_maps[site] && (
          <figure style={{ marginTop: 12 }}>
            {/* Plain <img>: the static export runs with images.unoptimized, so next/image
                would add machinery without optimising anything. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={data.site_maps[site]} alt={`Error maps for ${pretty(site)}`} style={{ width: '100%' }} />
            <figcaption className="muted small">{pretty(site)} — prediction, reference and error</figcaption>
          </figure>
        )}
      </section>

      <section className="panel">
        <h3 className="panel-heading">Does it survive satellite resolution?</h3>
        <p className="muted small">
          The same eight sites, with the photograph progressively coarsened to imitate a satellite
          sensor. Sharper imagery is easier; the question is how fast accuracy falls away.
        </p>
        <table className="metrics-table">
          <thead>
            <tr><th>Image resolution</th><th>RMSE</th><th>MAE</th><th>r</th><th>vs sharpest</th></tr>
          </thead>
          <tbody>
            {data.by_gsd.map((r) => {
              const base = data.by_gsd[0]?.rmse ?? r.rmse;
              const d = ((r.rmse - base) / base) * 100;
              return (
                <tr key={r.gsd}>
                  <td>{r.gsd} m per pixel{r.gsd === 0.25 ? ' (aerial)' : ''}</td>
                  <td>{fmt(r.rmse)} m</td>
                  <td>{fmt(r.mae)} m</td>
                  <td>{fmt(r.pearson_r)}</td>
                  <td className="muted">{d < 0.05 ? '—' : `+${d.toFixed(1)}%`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {g && (
        <section className="panel">
          <h3 className="panel-heading">What the model is good and bad at</h3>
          <table className="metrics-table">
            <thead><tr><th>Surface</th><th>RMSE</th><th>Average error</th><th>Bias</th></tr></thead>
            <tbody>
              {Object.entries(g.by_pixel_class)
                .sort((a, b) => a[1].mae - b[1].mae)
                .map(([k, v]) => (
                  <tr key={k}>
                    <td>{CLASS_LABEL[k] ?? pretty(k)}</td><td>{fmt(v.rmse)} m</td><td>{fmt(v.mae)} m</td><td>{fmt(v.bias)} m</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="muted small" style={{ marginTop: 8 }}>
            Ground, roads and low vegetation are sub-metre. Buildings and trees carry the error, and both
            are under-predicted (negative bias) — a single straight-down view has no parallax, so the
            height of a tall object has to be inferred rather than measured.
          </p>
        </section>
      )}
    </div>
  );
}
