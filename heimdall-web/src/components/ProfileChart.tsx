"use client";

import type { ProfilePoint } from '@/lib/terrain';

/** Elevation profile along a measured segment (surface + above-ground height). */
export default function ProfileChart({ points, units }: { points: ProfilePoint[]; units: string }) {
  const valid = points.filter((p) => Number.isFinite(p.h));
  if (valid.length < 2) return null;
  const W = 300, H = 110, L = 38, B = 18, T = 8, R = 6;
  const dMax = points[points.length - 1].d || 1;
  let hMin = Math.min(...valid.map((p) => p.h));
  let hMax = Math.max(...valid.map((p) => p.h));
  if (hMax - hMin < 1e-6) { hMax += 0.5; hMin -= 0.5; }
  const pad = (hMax - hMin) * 0.08;
  hMin -= pad; hMax += pad;
  const x = (d: number) => L + (d / dMax) * (W - L - R);
  const y = (h: number) => T + (1 - (h - hMin) / (hMax - hMin)) * (H - T - B);
  const path = valid.map((p, i) => `${i ? 'L' : 'M'}${x(p.d).toFixed(1)},${y(p.h).toFixed(1)}`).join(' ');
  const area = `${path} L${x(valid[valid.length - 1].d)},${H - B} L${x(valid[0].d)},${H - B} Z`;
  const ticks = [hMin + pad, (hMin + hMax) / 2, hMax - pad];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Elevation profile">
      <defs>
        <linearGradient id="pf" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="rgba(255,255,255,0.08)" />
          <text x={L - 4} y={y(t) + 3} fill="#9aa3b2" fontSize="8.5" textAnchor="end">{t.toFixed(1)}</text>
        </g>
      ))}
      <path d={area} fill="url(#pf)" />
      <path d={path} fill="none" stroke="#22d3ee" strokeWidth="1.6" />
      <text x={L} y={H - 4} fill="#9aa3b2" fontSize="8.5">0 m</text>
      <text x={W - R} y={H - 4} fill="#9aa3b2" fontSize="8.5" textAnchor="end">{dMax.toFixed(1)} m</text>
      <text x={4} y={T + 4} fill="#9aa3b2" fontSize="8.5">{units}</text>
    </svg>
  );
}
