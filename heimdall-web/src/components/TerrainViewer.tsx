"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { Html, Line, OrbitControls, useGLTF, useProgress } from '@react-three/drei';
import * as THREE from 'three';
import ProfileChart from '@/components/ProfileChart';
import type { Scene } from '@/lib/types';
import {
  NODATA, cssGradient, floodStats, fmt, localToCrs, localToLatLon, profile, sampleAt, slopeColor, turbo,
  valueToY, type TerrainGrid,
} from '@/lib/terrain';

type Mode = 'orbit' | 'measure' | 'fly';
type Layer = 'optical' | 'elevation' | 'agl' | 'slope';

interface Hover {
  x: number;
  z: number;
  h: number;
  agl: number;
  slope: number;
}

/* ─── Loading overlay ─────────────────────────────────────────────────── */
function Loading() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="viewer-loading">Loading terrain… {progress.toFixed(0)}%</div>
    </Html>
  );
}

/* ─── Terrain mesh ────────────────────────────────────────────────────── */
function Terrain({
  url, grid, layer, onHover, onPick,
}: {
  url: string;
  grid: TerrainGrid;
  layer: Layer;
  onHover: (p: THREE.Vector3 | null) => void;
  onPick: (p: THREE.Vector3) => void;
}) {
  const gltf = useGLTF(url);
  const { geometry, texMaterial } = useMemo(() => {
    let found: THREE.Mesh | null = null;
    gltf.scene.traverse((o) => {
      if (!found && (o as THREE.Mesh).isMesh) found = o as THREE.Mesh;
    });
    const mesh = found as unknown as THREE.Mesh;
    const geo = mesh.geometry as THREE.BufferGeometry;
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    const mat = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial;
    mat.side = THREE.DoubleSide;
    mat.roughness = 1;
    mat.metalness = 0;
    if (mat.map) mat.map.anisotropy = 8;
    // Nadir imagery has no façades: on near-vertical faces the texture is smeared, so blend it
    // towards a neutral wall tone by how vertical the face is.
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWorldNormalH;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n vWorldNormalH = normalize(mat3(modelMatrix) * objectNormal);');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWorldNormalH;')
        .replace('#include <map_fragment>', `#include <map_fragment>
          float wallness = smoothstep(0.55, 0.18, abs(vWorldNormalH.y));
          vec3 wallTone = vec3(dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114))) * 0.82;
          diffuseColor.rgb = mix(diffuseColor.rgb, wallTone, wallness * 0.85);`);
    };
    mat.needsUpdate = true;
    return { geometry: geo, texMaterial: mat };
  }, [gltf]);

  const colorMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0, side: THREE.DoubleSide }),
    [],
  );

  useEffect(() => {
    if (layer === 'optical') return;
    const key = layer === 'elevation' ? 'surface' : layer === 'agl' ? 'ndsm' : 'slope_deg';
    const values = grid.layers[key];
    if (!values) return;
    const n = geometry.getAttribute('position').count;
    const colors = new Float32Array(n * 3);
    const st = grid.stats[key];
    const lo = layer === 'agl' ? 0 : st.p02;
    const hi = Math.max(st.p98, lo + 1e-6);
    for (let i = 0; i < n; i++) {
      const v = values[i];
      let c: [number, number, number];
      if (v === NODATA || !Number.isFinite(v)) c = [0.15, 0.15, 0.18];
      else if (layer === 'slope') c = slopeColor(v);
      else c = turbo((v - lo) / (hi - lo));
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }, [layer, grid, geometry]);

  return (
    <mesh
      geometry={geometry}
      material={layer === 'optical' ? texMaterial : colorMaterial}
      castShadow
      receiveShadow
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        onHover(e.point.clone());
      }}
      onPointerOut={() => onHover(null)}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.delta < 4) onPick(e.point.clone());
      }}
    />
  );
}

/* ─── Water plane for flood simulation ────────────────────────────────── */
function Water({ grid, level }: { grid: TerrainGrid; level: number }) {
  const y = valueToY(grid, level);
  return (
    <mesh position={[0, y, 0]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={2}>
      <planeGeometry args={[grid.widthM, grid.depthM]} />
      <meshPhysicalMaterial color="#1d6fd1" transparent opacity={0.55} roughness={0.15} metalness={0.1}
        depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

/* ─── Camera framing ──────────────────────────────────────────────────── */
function FitCamera({ grid, resetKey }: { grid: TerrainGrid; resetKey: number }) {
  const get = useThree((st) => st.get);
  const controls = useThree((st) => st.controls);
  useEffect(() => {
    const { camera, scene } = get();
    const D = Math.max(grid.widthM, grid.depthM);
    const cam = camera as THREE.PerspectiveCamera;
    cam.near = Math.max(0.05, D / 5000);
    cam.far = D * 60;
    cam.position.set(D * 0.05, D * 0.62, D * 0.78);
    cam.updateProjectionMatrix();
    const oc = controls as unknown as { target: THREE.Vector3; update: () => void } | null;
    if (oc) {
      oc.target.set(0, 0, 0);
      oc.update();
    } else {
      cam.lookAt(0, 0, 0);
    }
    scene.fog = new THREE.Fog('#070a10', D * 2.5, D * 12);
  }, [grid, resetKey, controls, get]);
  return null;
}

/* ─── First-person flythrough ─────────────────────────────────────────── */
function FlyControls({ active, grid, exaggeration }: { active: boolean; grid: TerrainGrid; exaggeration: number }) {
  const { camera, gl } = useThree();
  const keys = useRef<Set<string>>(new Set());
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const locked = useRef(false);

  useEffect(() => {
    if (!active) return;
    const typing = (e: KeyboardEvent) => (e.target as HTMLElement)?.closest?.('input, textarea, select');
    const down = (e: KeyboardEvent) => {
      if (typing(e)) return;
      keys.current.add(e.code);
      if (['Space', 'ArrowUp', 'ArrowDown'].includes(e.code)) e.preventDefault();
    };
    const up = (e: KeyboardEvent) => keys.current.delete(e.code);
    const move = (e: MouseEvent) => {
      if (!locked.current) return;
      euler.current.setFromQuaternion(camera.quaternion);
      euler.current.y -= e.movementX * 0.0022;
      euler.current.x = Math.max(-1.5, Math.min(1.5, euler.current.x - e.movementY * 0.0022));
      camera.quaternion.setFromEuler(euler.current);
    };
    const click = () => gl.domElement.requestPointerLock?.();
    const lockChange = () => {
      locked.current = document.pointerLockElement === gl.domElement;
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    document.addEventListener('mousemove', move);
    gl.domElement.addEventListener('click', click);
    document.addEventListener('pointerlockchange', lockChange);
    const pressed = keys.current;
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      document.removeEventListener('mousemove', move);
      gl.domElement.removeEventListener('click', click);
      document.removeEventListener('pointerlockchange', lockChange);
      if (document.pointerLockElement === gl.domElement) document.exitPointerLock();
      pressed.clear();
    };
  }, [active, camera, gl]);

  useFrame((state, dt) => {
    if (!active) return;
    const cam = state.camera;
    const k = keys.current;
    const D = Math.max(grid.widthM, grid.depthM);
    const speed = (D / 12) * (k.has('ShiftLeft') || k.has('ShiftRight') ? 4 : 1) * Math.min(dt, 0.05);
    const dir = new THREE.Vector3(
      (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0),
      (k.has('Space') || k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') || k.has('KeyC') ? 1 : 0),
      (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0) - (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0),
    );
    if (dir.lengthSq() > 0) {
      const vertical = dir.y;
      dir.y = 0;
      dir.normalize().applyQuaternion(cam.quaternion);
      cam.position.addScaledVector(dir, speed);
      cam.position.y += vertical * speed;
    }
    // Terrain collision: never go below ground + clearance.
    const ground = sampleAt(grid, 'surface', cam.position.x, cam.position.z);
    if (Number.isFinite(ground)) {
      const minY = valueToY(grid, ground) * exaggeration + Math.max(1.5, D / 400);
      if (cam.position.y < minY) cam.position.y = minY;
    }
  });
  return null;
}

/* ─── Screen pixels per ground metre, for the scale bar ───────────────── */
function ScaleProbe({ onChange }: { onChange: (v: number) => void }) {
  const last = useRef(0);
  useFrame((state) => {
    const cam = state.camera as THREE.PerspectiveCamera;
    if (!cam.isPerspectiveCamera) return;
    // Metres per pixel at the distance of the scene centre, then invert.
    const dist = cam.position.length();
    const worldPerPx = (2 * Math.tan((cam.fov * Math.PI) / 360) * dist) / state.size.height;
    const v = 1 / Math.max(worldPerPx, 1e-9);
    if (Math.abs(v - last.current) / Math.max(v, 1e-9) > 0.02) {
      last.current = v;
      onChange(v);
    }
  });
  return null;
}

/* ─── Screenshot helper ───────────────────────────────────────────────── */
function Capture({ onReady }: { onReady: (fn: () => string) => void }) {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    onReady(() => {
      gl.render(scene, camera);
      return gl.domElement.toDataURL('image/png');
    });
  }, [gl, scene, camera, onReady]);
  return null;
}

/* ─── Main viewer ─────────────────────────────────────────────────────── */
export default function TerrainViewer({ scene, grid }: { scene: Scene; grid: TerrainGrid | null }) {
  const meta = scene.meta;
  const units = meta.units === 'm' ? 'm' : 'rel';
  const hasAgl = !!grid?.layers.ndsm;
  const [mode, setMode] = useState<Mode>('orbit');
  const [layer, setLayer] = useState<Layer>('optical');
  const [exaggeration, setExaggeration] = useState(1);
  const [autoRotate, setAutoRotate] = useState(false);
  const [sunAz, setSunAz] = useState(315);
  const [flood, setFlood] = useState(false);
  const [level, setLevel] = useState<number | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [points, setPoints] = useState<THREE.Vector3[]>([]);
  const [resetKey, setResetKey] = useState(0);
  const snap = useRef<(() => string) | null>(null);

  const meshUrl = scene.baseUrl + meta.files.mesh;
  const sStats = grid?.stats.surface;
  const floodLevel = level ?? (sStats ? sStats.p02 + (sStats.p98 - sStats.p02) * 0.15 : 0);
  const floodInfo = useMemo(() => (grid && flood ? floodStats(grid, floodLevel) : null), [grid, flood, floodLevel]);

  const toLocal = useCallback((p: THREE.Vector3) => new THREE.Vector3(p.x, p.y / exaggeration, p.z), [exaggeration]);

  const onHover = useCallback(
    (p: THREE.Vector3 | null) => {
      if (!p || !grid) return setHover(null);
      const l = toLocal(p);
      setHover({
        x: l.x, z: l.z,
        h: sampleAt(grid, 'surface', l.x, l.z),
        agl: sampleAt(grid, 'ndsm', l.x, l.z),
        slope: sampleAt(grid, 'slope_deg', l.x, l.z),
      });
    },
    [grid, toLocal],
  );

  const onPick = useCallback(
    (p: THREE.Vector3) => {
      if (mode !== 'measure') return;
      const l = toLocal(p);
      setPoints((prev) => (prev.length >= 2 ? [l] : [...prev, l]));
    },
    [mode, toLocal],
  );


  const measurement = useMemo(() => {
    if (!grid || points.length < 2) return null;
    const [a, b] = points;
    const ha = sampleAt(grid, 'surface', a.x, a.z);
    const hb = sampleAt(grid, 'surface', b.x, b.z);
    const horiz = Math.hypot(b.x - a.x, b.z - a.z);
    const dh = hb - ha;
    const dist3d = units === 'm' ? Math.hypot(horiz, dh) : NaN;
    const prof = profile(grid, [a.x, a.z], [b.x, b.z]);
    return { ha, hb, horiz, dh, dist3d, grade: units === 'm' ? (dh / horiz) * 100 : NaN, prof };
  }, [grid, points, units]);

  const sun = useMemo(() => {
    const D = grid ? Math.max(grid.widthM, grid.depthM) : 100;
    const az = (sunAz * Math.PI) / 180;
    return new THREE.Vector3(Math.sin(az) * D, D * 0.9, -Math.cos(az) * D);
  }, [sunAz, grid]);

  const world = (l: THREE.Vector3) => new THREE.Vector3(l.x, l.y * exaggeration, l.z);
  const D = grid ? Math.max(grid.widthM, grid.depthM) : 100;
  const legend = layer === 'optical' ? null : layer === 'slope'
    ? { grad: cssGradient('slope'), lo: '0°', hi: '≥50°', title: 'Slope' }
    : layer === 'agl'
      ? { grad: cssGradient('turbo'), lo: fmt(0, 1), hi: fmt(grid?.stats.ndsm?.p98, 1), title: 'Height above ground (m)' }
      : {
          grad: cssGradient('turbo'),
          lo: fmt(grid?.stats.surface?.p02, 1),
          hi: fmt(grid?.stats.surface?.p98, 1),
          title: `Elevation (${units})`,
        };
  // Scale bar: a "nice" round distance and how wide it is on screen at the current camera.
  const [pxPerMetre, setPxPerMetre] = useState(0);
  const scaleBar = useMemo(() => {
    if (!pxPerMetre || !Number.isFinite(pxPerMetre)) return null;
    const target = 140 / pxPerMetre; // aim for ~140 px
    const pow = Math.pow(10, Math.floor(Math.log10(Math.max(target, 1e-6))));
    const nice = [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= target * 0.6) ?? pow * 10;
    const px = nice * pxPerMetre;
    if (!Number.isFinite(px) || px < 20 || px > 420) return null;
    return { px, label: nice >= 1000 ? `${nice / 1000} km` : `${nice} m` };
  }, [pxPerMetre]);

  const crs = hover && grid ? localToCrs(meta, grid, hover.x, hover.z) : null;
  const ll = hover && grid ? localToLatLon(meta, grid, hover.x, hover.z) : null;

  const screenshot = () => {
    const url = snap.current?.();
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = `heimdall_${scene.id}.png`;
    a.click();
  };

  const modes: { key: Mode; label: string; hint: string }[] = [
    { key: 'orbit', label: 'Orbit', hint: 'Drag to rotate · right-drag to pan · scroll to zoom · hover to read heights' },
    { key: 'measure', label: 'Measure', hint: 'Click two points to measure distance, height difference and profile' },
    { key: 'fly', label: 'Fly', hint: 'Click to capture mouse · WASD/arrows move · Space/E up · Q/C down · Shift boost · Esc release' },
  ];

  return (
    <div className="viewer">
      <div className="viewer-toolbar">
        {modes.map((m) => (
          <button key={m.key} className={`chip ${mode === m.key ? 'chip--on' : ''}`}
            onClick={() => { setMode(m.key); if (m.key !== 'measure') setPoints([]); }}>
            {m.label}
          </button>
        ))}
        <span className="toolbar-sep" />
        {(['optical', 'elevation', ...(hasAgl ? ['agl'] : []), 'slope'] as Layer[]).map((l) => (
          <button key={l} className={`chip chip--layer ${layer === l ? 'chip--on' : ''}`} onClick={() => setLayer(l)}>
            {{ optical: 'Optical', elevation: 'Elevation', agl: 'Above ground', slope: 'Slope' }[l]}
          </button>
        ))}
      </div>

      <div className="viewer-hint">{modes.find((m) => m.key === mode)?.hint}</div>

      <div className="viewer-controls">
        <label className="ctl">
          <span>Vertical exaggeration <b>{exaggeration.toFixed(1)}×</b></span>
          <input type="range" min={1} max={10} step={0.5} value={exaggeration}
            onChange={(e) => setExaggeration(parseFloat(e.target.value))} />
        </label>
        <label className="ctl">
          <span>Sun azimuth <b>{sunAz}°</b></span>
          <input type="range" min={0} max={360} step={5} value={sunAz} onChange={(e) => setSunAz(parseInt(e.target.value))} />
        </label>
        <label className="ctl ctl--check">
          <input type="checkbox" checked={flood} onChange={(e) => setFlood(e.target.checked)} /> Flood / water level
        </label>
        {flood && sStats && (
          <label className="ctl">
            <span>Water level <b>{fmt(floodLevel, 2, units)}</b></span>
            <input type="range" min={sStats.min} max={sStats.max} step={(sStats.max - sStats.min) / 400 || 0.01}
              value={floodLevel} onChange={(e) => setLevel(parseFloat(e.target.value))} />
            {floodInfo && (
              <small>
                Inundated: <b>{(floodInfo.fraction * 100).toFixed(1)}%</b>
                {units === 'm' && <> · {(floodInfo.areaM2 / 1e4).toFixed(2)} ha</>}
              </small>
            )}
          </label>
        )}
        <div className="ctl-row">
          <label className="ctl ctl--check">
            <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} /> Auto-rotate
          </label>
          <button className="chip" onClick={() => setResetKey((k) => k + 1)}>Reset view</button>
          <button className="chip" onClick={screenshot}>Screenshot</button>
        </div>
      </div>

      {hover && Number.isFinite(hover.h) && (
        <div className="viewer-hud">
          <div><span>Elevation</span><b>{fmt(hover.h, 2, units)}</b></div>
          {hasAgl && <div><span>Above ground</span><b>{fmt(hover.agl, 2, 'm')}</b></div>}
          <div><span>Slope</span><b>{fmt(hover.slope, 1, '°')}</b></div>
          {crs && <div><span>E / N{meta.crs_epsg ? ` (EPSG:${meta.crs_epsg})` : ''}</span><b>{crs[0].toFixed(2)}, {crs[1].toFixed(2)}</b></div>}
          {ll && <div><span>Lat / Lon ≈</span><b>{ll[0].toFixed(6)}, {ll[1].toFixed(6)}</b></div>}
          {!crs && <div><span>Local x / y</span><b>{hover.x.toFixed(1)}, {(-hover.z).toFixed(1)} m</b></div>}
        </div>
      )}

      {mode === 'measure' && (
        <div className="viewer-measure">
          {!measurement ? (
            <p>{points.length === 0 ? 'Click the first point on the terrain.' : 'Click the second point.'}</p>
          ) : (
            <>
              <div className="measure-grid">
                <div><span>Horizontal</span><b>{fmt(measurement.horiz, 2, 'm')}</b></div>
                <div><span>Δ height</span><b>{fmt(measurement.dh, 2, units)}</b></div>
                {units === 'm' && <div><span>3-D distance</span><b>{fmt(measurement.dist3d, 2, 'm')}</b></div>}
                {units === 'm' && <div><span>Grade</span><b>{fmt(measurement.grade, 1, '%')}</b></div>}
                <div><span>A</span><b>{fmt(measurement.ha, 2, units)}</b></div>
                <div><span>B</span><b>{fmt(measurement.hb, 2, units)}</b></div>
              </div>
              <ProfileChart points={measurement.prof} units={units} />
            </>
          )}
        </div>
      )}

      {grid && (
        <div className="viewer-compass" aria-label="North indicator">
          <svg width="46" height="46" viewBox="0 0 46 46">
            <circle cx="23" cy="23" r="20" fill="rgba(8,11,17,0.75)" stroke="rgba(255,255,255,0.15)" />
            <polygon points="23,7 28,25 23,21 18,25" fill="#f87171" />
            <polygon points="23,39 18,21 23,25 28,21" fill="#cbd5e1" />
            <text x="23" y="6" fontSize="7" fill="#cbd5e1" textAnchor="middle">N</text>
          </svg>
        </div>
      )}

      {grid && scaleBar && (
        <div className="viewer-scalebar" aria-label="Scale bar">
          <div className="scalebar-line" style={{ width: `${scaleBar.px}px` }} />
          <span>{scaleBar.label}</span>
        </div>
      )}

      {legend && (
        <div className="viewer-legend">
          <div className="legend-title">{legend.title}</div>
          <div className="legend-bar" style={{ background: legend.grad }} />
          <div className="legend-labels"><span>{legend.lo}</span><span>{legend.hi}</span></div>
        </div>
      )}

      <Canvas shadows gl={{ preserveDrawingBuffer: true, antialias: true }} camera={{ fov: 50 }}
        onPointerMissed={() => setHover(null)}>
        <color attach="background" args={['#070a10']} />
        <hemisphereLight args={['#dfe9ff', '#2a2418', 0.85]} />
        <directionalLight position={sun} intensity={1.6} castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-D * 0.8} shadow-camera-right={D * 0.8}
          shadow-camera-top={D * 0.8} shadow-camera-bottom={-D * 0.8}
          shadow-camera-far={D * 4} shadow-bias={-0.0005} />
        {grid && (
          <Suspense fallback={<Loading />}>
            <group scale={[1, exaggeration, 1]}>
              <Terrain url={meshUrl} grid={grid} layer={layer} onHover={onHover} onPick={onPick} />
              {flood && <Water grid={grid} level={floodLevel} />}
            </group>
            <FitCamera grid={grid} resetKey={resetKey} />
          </Suspense>
        )}
        {points.map((p, i) => (
          <mesh key={i} position={world(p)}>
            <sphereGeometry args={[D / 250, 16, 16]} />
            <meshBasicMaterial color={i === 0 ? '#22d3ee' : '#f472b6'} depthTest={false} />
          </mesh>
        ))}
        {points.length === 2 && (
          <Line points={[world(points[0]), world(points[1])]} color="#fbbf24" lineWidth={2.5} depthTest={false} />
        )}
        {mode !== 'fly' && (
          <OrbitControls makeDefault autoRotate={autoRotate && mode === 'orbit'} autoRotateSpeed={0.6}
            enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI * 0.495} minDistance={D / 200} maxDistance={D * 8} />
        )}
        {grid && <FlyControls active={mode === 'fly'} grid={grid} exaggeration={exaggeration} />}
        <ScaleProbe onChange={setPxPerMetre} />
        <Capture onReady={(fn) => { snap.current = fn; }} />
      </Canvas>
    </div>
  );
}
