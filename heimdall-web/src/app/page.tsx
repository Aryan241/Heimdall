import React from 'react';
import { UploadCloud, Layers, Zap, Hexagon, HardDrive } from 'lucide-react';
import Viewer3D from '@/components/Viewer3D';

export default function Home() {
  return (
    <div className="container animate-fade-in">
      <header className="header">
        <div className="logo text-gradient">
          <Hexagon size={28} color="var(--accent-cyan)" />
          HEIMDALL
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Monocular Depth & Mesh Pipeline
        </div>
      </header>

      <main className="main-layout">
        <aside className="dashboard-sidebar">
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>New Processing Job</h3>
            
            <div className="upload-zone">
              <UploadCloud size={48} className="upload-icon" />
              <p style={{ fontWeight: 600 }}>Drop Satellite Image Here</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                JPEG or PNG, up to 100MB
              </p>
            </div>
            
            <div className="upload-zone" style={{ marginTop: '1rem' }}>
              <Layers size={48} className="upload-icon" />
              <p style={{ fontWeight: 600 }}>Drop Reference DEM (Optional)</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                For absolute scale calibration
              </p>
            </div>

            <button className="btn-primary" style={{ marginTop: '1.5rem' }}>
              <Zap size={16} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} />
              Process Pipeline
            </button>
          </div>

          <div className="glass-panel" style={{ padding: '1.5rem', flexGrow: 1 }}>
            <h3 style={{ marginBottom: '1.5rem', color: 'var(--text-primary)' }}>Pipeline Status</h3>
            <ul className="status-list">
              <li className="status-item">
                <div className="status-icon done">✓</div>
                <div>
                  <strong>Stage 1: Ingestion</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Image parsed successfully</div>
                </div>
              </li>
              <li className="status-item">
                <div className="status-icon done">✓</div>
                <div>
                  <strong>Stage 4: Segmentation</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Segformer-B0 complete</div>
                </div>
              </li>
              <li className="status-item">
                <div className="status-icon done">✓</div>
                <div>
                  <strong>Stage 2: Depth Inference</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Depth Anything V2 complete</div>
                </div>
              </li>
              <li className="status-item">
                <div className="status-icon done">✓</div>
                <div>
                  <strong>Stage 5: RANSAC Calibration</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Scale calibrated to 10m</div>
                </div>
              </li>
              <li className="status-item">
                <div className="status-icon active">
                  <HardDrive size={14} />
                </div>
                <div>
                  <strong style={{ color: 'var(--accent-cyan)' }}>Stage 8: Meshing</strong>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Rendering PLY preview...</div>
                </div>
              </li>
            </ul>
          </div>
        </aside>

        <section className="canvas-container glass-panel">
          {/* We load the test_aerial_mesh.ply that we copied to public/models/ */}
          <Viewer3D modelUrl="/models/test_aerial_mesh.ply" />
          
          <div style={{ position: 'absolute', bottom: '20px', left: '20px', background: 'rgba(0,0,0,0.5)', padding: '10px 15px', borderRadius: '8px', backdropFilter: 'blur(8px)', border: '1px solid var(--border-glass)' }}>
            <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>3D Mesh Preview</h4>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Interactive Orbit View</p>
          </div>
        </section>
      </main>
    </div>
  );
}
