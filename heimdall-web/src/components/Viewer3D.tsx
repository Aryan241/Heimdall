"use client";

import { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stage, useProgress, Html } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import * as THREE from 'three';

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div style={{ color: 'white', fontFamily: 'monospace', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '8px' }}>
        {progress.toFixed(1)} % loaded
      </div>
    </Html>
  );
}

function PlyModel({ url }: { url: string }) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    const loader = new PLYLoader();
    loader.load(url, (geo) => {
      geo.computeVertexNormals();
      setGeometry(geo);
    });
  }, [url]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors={true} side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function Viewer3D({ modelUrl }: { modelUrl: string }) {
  return (
    <Canvas shadows camera={{ position: [0, 40, 150], fov: 45 }}>
      <color attach="background" args={['#050505']} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[100, 100, 50]} intensity={1.5} castShadow />
      
      {/* We rotate the mesh -90 degrees on the X axis because Python exports Z-up meshes, but Three.js uses Y-up */}
      <group position={[0, -20, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <PlyModel url={modelUrl} />
      </group>
      
      {/* 
        OrbitControls autoRotate provides a cinematic flythrough effect 
        but instantly gives control back to the user when they click/drag!
      */}
      <OrbitControls 
        makeDefault 
        autoRotate 
        autoRotateSpeed={1.5} 
        enableZoom={true} 
        enablePan={true} 
      />
    </Canvas>
  );
}
