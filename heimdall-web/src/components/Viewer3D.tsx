"use client";

import { useEffect, useState, useRef, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, useProgress, Html, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

/* ─── Loading indicator ──────────────────────────────────────────────── */
function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div style={{
        color: 'white',
        fontFamily: 'monospace',
        background: 'rgba(0,0,0,0.7)',
        padding: '16px 24px',
        borderRadius: '12px',
        backdropFilter: 'blur(8px)',
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        {progress.toFixed(1)} % loaded
      </div>
    </Html>
  );
}

/* ─── Slope Heatmap Shader ───────────────────────────────────────────── */
const slopeMaterial = new THREE.ShaderMaterial({
  vertexShader: `
    varying vec3 vLocalNormal;
    void main() {
      vLocalNormal = normal;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec3 vLocalNormal;
    void main() {
      float upDot = abs(vLocalNormal.z);
      float slope = 1.0 - upDot;
      vec3 flatColor = vec3(0.1, 0.8, 0.3);
      vec3 midColor = vec3(0.9, 0.8, 0.1);
      vec3 steepColor = vec3(0.9, 0.1, 0.1);
      vec3 color = mix(flatColor, midColor, smoothstep(0.0, 0.5, slope));
      color = mix(color, steepColor, smoothstep(0.5, 1.0, slope));
      gl_FragColor = vec4(color, 1.0);
    }
  `,
  side: THREE.DoubleSide,
});

/* ─── Height Colormap Shader ─────────────────────────────────────────── */
const heightMaterial = new THREE.ShaderMaterial({
  vertexShader: `
    varying float vHeight;
    uniform float uMinHeight;
    uniform float uMaxHeight;
    void main() {
      vHeight = position.z;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying float vHeight;
    uniform float uMinHeight;
    uniform float uMaxHeight;
    void main() {
      float t = clamp((vHeight - uMinHeight) / (uMaxHeight - uMinHeight), 0.0, 1.0);
      // Turbo-ish colormap: blue → cyan → green → yellow → red
      vec3 c;
      if (t < 0.25) {
        c = mix(vec3(0.18, 0.19, 0.57), vec3(0.09, 0.62, 0.82), t * 4.0);
      } else if (t < 0.5) {
        c = mix(vec3(0.09, 0.62, 0.82), vec3(0.32, 0.85, 0.27), (t - 0.25) * 4.0);
      } else if (t < 0.75) {
        c = mix(vec3(0.32, 0.85, 0.27), vec3(0.95, 0.82, 0.11), (t - 0.5) * 4.0);
      } else {
        c = mix(vec3(0.95, 0.82, 0.11), vec3(0.84, 0.15, 0.16), (t - 0.75) * 4.0);
      }
      gl_FragColor = vec4(c, 1.0);
    }
  `,
  uniforms: {
    uMinHeight: { value: 0 },
    uMaxHeight: { value: 100 },
  },
  side: THREE.DoubleSide,
});

/* ─── PLY Mesh Component ─────────────────────────────────────────────── */
function PlyModel({
  url,
  mode,
  onHeightInfo,
}: {
  url: string;
  mode: 'fly' | 'height' | 'slope' | 'colormap';
  onHeightInfo: (info: { height: number; point: THREE.Vector3 } | null) => void;
}) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  // Load GLB using drei's useGLTF
  const { scene } = useGLTF(url);

  useEffect(() => {
    // Find the first mesh in the scene
    let mesh: THREE.Mesh | null = null;
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh && !mesh) {
        mesh = child as THREE.Mesh;
      }
    });

    if (mesh) {
      const geo = mesh.geometry;
      geo.computeVertexNormals();
      geo.computeBoundingBox();

      if (geo.boundingBox) {
        heightMaterial.uniforms.uMinHeight.value = geo.boundingBox.min.z;
        heightMaterial.uniforms.uMaxHeight.value = geo.boundingBox.max.z;
      }
      setGeometry(geo);
    }
  }, [scene]);

  if (!geometry) return <Loader />;

  const handlePointerMove = (e: any) => {
    if (mode === 'height') {
      e.stopPropagation();
      // The mesh is rotated -90deg on X, so the "Z" in local space is the height.
      // But the intersection point is in world space where Y is up after rotation.
      // We need the LOCAL z-coordinate for the height value.
      const localPoint = e.object.worldToLocal(e.point.clone());
      onHeightInfo({
        height: localPoint.z,
        point: e.point.clone(),
      });
    }
  };

  const handlePointerLeave = () => {
    if (mode === 'height') {
      onHeightInfo(null);
    }
  };

  const getMaterial = () => {
    switch (mode) {
      case 'slope':
        return <primitive object={slopeMaterial} attach="material" />;
      case 'colormap':
        return <primitive object={heightMaterial} attach="material" />;
      default:
        return <meshStandardMaterial vertexColors={true} side={THREE.DoubleSide} />;
    }
  };

  return (
    <mesh
      geometry={geometry}
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      {getMaterial()}
    </mesh>
  );
}

/* ─── WASD First-Person Controls ─────────────────────────────────────── */
function FPSControls({ active }: { active: boolean }) {
  const { camera, gl } = useThree();
  const keys = useRef<Set<string>>(new Set());
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isPointerLocked = useRef(false);

  useEffect(() => {
    if (!active) return;

    const handleKeyDown = (e: KeyboardEvent) => keys.current.add(e.code);
    const handleKeyUp = (e: KeyboardEvent) => keys.current.delete(e.code);

    const handleMouseMove = (e: MouseEvent) => {
      if (!isPointerLocked.current) return;
      euler.current.setFromQuaternion(camera.quaternion);
      euler.current.y -= e.movementX * 0.002;
      euler.current.x -= e.movementY * 0.002;
      euler.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, euler.current.x));
      camera.quaternion.setFromEuler(euler.current);
    };

    const handleClick = () => {
      gl.domElement.requestPointerLock();
    };

    const handleLockChange = () => {
      isPointerLocked.current = document.pointerLockElement === gl.domElement;
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    document.addEventListener('mousemove', handleMouseMove);
    gl.domElement.addEventListener('click', handleClick);
    document.addEventListener('pointerlockchange', handleLockChange);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      document.removeEventListener('mousemove', handleMouseMove);
      gl.domElement.removeEventListener('click', handleClick);
      document.removeEventListener('pointerlockchange', handleLockChange);
      if (document.pointerLockElement === gl.domElement) {
        document.exitPointerLock();
      }
    };
  }, [active, camera, gl]);

  useFrame((_, delta) => {
    if (!active || !isPointerLocked.current) return;

    const speed = 80 * delta;
    const direction = new THREE.Vector3();

    if (keys.current.has('KeyW')) direction.z -= 1;
    if (keys.current.has('KeyS')) direction.z += 1;
    if (keys.current.has('KeyA')) direction.x -= 1;
    if (keys.current.has('KeyD')) direction.x += 1;
    if (keys.current.has('Space')) direction.y += 1;
    if (keys.current.has('ShiftLeft')) direction.y -= 1;

    direction.normalize();
    direction.applyQuaternion(camera.quaternion);
    camera.position.add(direction.multiplyScalar(speed));
  });

  return null;
}

/* ─── Height Tooltip Component ───────────────────────────────────────── */
function HeightTooltip({ info }: { info: { height: number; point: THREE.Vector3 } | null }) {
  if (!info) return null;

  return (
    <Html position={info.point} center style={{ pointerEvents: 'none' }}>
      <div style={{
        background: 'rgba(0, 0, 0, 0.85)',
        color: '#00ff88',
        padding: '8px 14px',
        borderRadius: '8px',
        fontWeight: 'bold',
        fontFamily: 'monospace',
        fontSize: '14px',
        whiteSpace: 'nowrap',
        border: '1px solid rgba(0, 255, 136, 0.3)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        backdropFilter: 'blur(4px)',
      }}>
        ▲ {info.height.toFixed(2)} m
      </div>
    </Html>
  );
}

/* ─── Mode Button SVG Icons ──────────────────────────────────────────── */
function PlaneIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
    </svg>
  );
}

function CrosshairIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><line x1="22" y1="12" x2="18" y2="12" />
      <line x1="6" y1="12" x2="2" y2="12" /><line x1="12" y1="6" x2="12" y2="2" />
      <line x1="12" y1="22" x2="12" y2="18" />
    </svg>
  );
}

function MountainIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m8 3 4 8 5-5 5 15H2L8 3z" />
    </svg>
  );
}

function GamepadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="6" y1="11" x2="10" y2="11" /><line x1="8" y1="9" x2="8" y2="13" />
      <line x1="15" y1="12" x2="15.01" y2="12" /><line x1="18" y1="10" x2="18.01" y2="10" />
      <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z" />
    </svg>
  );
}

function PaletteIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="13.5" cy="6.5" r=".5" fill="currentColor" /><circle cx="17.5" cy="10.5" r=".5" fill="currentColor" />
      <circle cx="8.5" cy="7.5" r=".5" fill="currentColor" /><circle cx="6.5" cy="12.5" r=".5" fill="currentColor" />
      <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z" />
    </svg>
  );
}

/* ─── Main Viewer3D Component ────────────────────────────────────────── */
export default function Viewer3D({ modelUrl }: { modelUrl: string }) {
  const [mode, setMode] = useState<'fly' | 'height' | 'slope' | 'fps' | 'colormap'>('fly');
  const [heightInfo, setHeightInfo] = useState<{ height: number; point: THREE.Vector3 } | null>(null);

  const modeConfig = [
    { key: 'fly' as const, label: 'Orbit', icon: <PlaneIcon />, color: '#6366f1', glow: 'rgba(99,102,241,0.5)' },
    { key: 'height' as const, label: 'Height', icon: <CrosshairIcon />, color: '#10b981', glow: 'rgba(16,185,129,0.5)' },
    { key: 'slope' as const, label: 'Slope', icon: <MountainIcon />, color: '#e11d48', glow: 'rgba(225,29,72,0.5)' },
    { key: 'colormap' as const, label: 'Elevation', icon: <PaletteIcon />, color: '#f59e0b', glow: 'rgba(245,158,11,0.5)' },
    { key: 'fps' as const, label: 'FPS', icon: <GamepadIcon />, color: '#8b5cf6', glow: 'rgba(139,92,246,0.5)' },
  ];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0a0f', overflow: 'hidden' }}>
      {/* ─── Mode Toolbar ─── */}
      <div style={{
        position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)', zIndex: 10,
        display: 'flex', gap: '4px', padding: '6px',
        background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(12px)',
        borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      }}>
        {modeConfig.map(({ key, label, icon, color, glow }) => (
          <button
            key={key}
            onClick={() => { setMode(key); setHeightInfo(null); }}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '8px 16px', borderRadius: '12px',
              border: 'none', cursor: 'pointer',
              transition: 'all 0.2s ease',
              background: mode === key ? color : 'transparent',
              color: mode === key ? 'white' : '#999',
              boxShadow: mode === key ? `0 0 15px ${glow}` : 'none',
              fontSize: '13px', fontWeight: 500, fontFamily: 'Inter, sans-serif',
            }}
          >
            {icon}
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* ─── FPS Mode Hint ─── */}
      {mode === 'fps' && (
        <div style={{
          position: 'absolute', bottom: '24px', left: '50%', transform: 'translateX(-50%)', zIndex: 10,
          color: '#a78bfa', fontWeight: 500, fontSize: '13px', fontFamily: 'Inter, sans-serif',
          background: 'rgba(0,0,0,0.6)', padding: '8px 16px', borderRadius: '10px',
          border: '1px solid rgba(139,92,246,0.2)', backdropFilter: 'blur(8px)',
        }}>
          Click to lock mouse · WASD to move · Space/Shift for up/down · Esc to release
        </div>
      )}

      {/* ─── Height Mode Hint ─── */}
      {mode === 'height' && !heightInfo && (
        <div style={{
          position: 'absolute', bottom: '24px', left: '50%', transform: 'translateX(-50%)', zIndex: 10,
          color: '#34d399', fontWeight: 500, fontSize: '13px', fontFamily: 'Inter, sans-serif',
          background: 'rgba(0,0,0,0.6)', padding: '8px 16px', borderRadius: '10px',
          border: '1px solid rgba(16,185,129,0.2)', backdropFilter: 'blur(8px)',
        }}>
          Hover over terrain to inspect structural height values
        </div>
      )}

      {/* ─── WebGL Canvas ─── */}
      <Canvas shadows camera={{ position: [0, 80, 200], fov: 45 }}>
        <color attach="background" args={['#0a0a0f']} />
        <fog attach="fog" args={['#0a0a0f', 300, 800]} />

        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 80]} intensity={1.2} castShadow />
        <directionalLight position={[-80, 60, -50]} intensity={0.3} />

        {/* Mesh group — rotate Z-up (Python) to Y-up (Three.js) */}
        <group position={[0, -20, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <PlyModel
            url={modelUrl}
            mode={mode === 'fps' ? 'fly' : mode}
            onHeightInfo={setHeightInfo}
          />
        </group>

        <HeightTooltip info={heightInfo} />

        {/* Orbit controls for non-FPS modes */}
        {mode !== 'fps' && (
          <OrbitControls
            makeDefault
            autoRotate={mode === 'fly'}
            autoRotateSpeed={1.0}
            enableZoom={true}
            enablePan={true}
            maxPolarAngle={Math.PI * 0.85}
          />
        )}

        {/* FPS controls */}
        <FPSControls active={mode === 'fps'} />
      </Canvas>
    </div>
  );
}
