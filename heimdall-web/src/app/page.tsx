"use client";

import React, { useState, useRef } from 'react';
import { UploadCloud, Layers, Zap, Hexagon, HardDrive, CheckCircle, Loader } from 'lucide-react';
import Viewer3D from '@/components/Viewer3D';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<string>('idle'); // idle, uploading, processing, done, error
  const [modelUrl, setModelUrl] = useState<string>('/models/stadium_mesh.ply');
  const [errorMsg, setErrorMsg] = useState<string>('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileClick = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus('idle');
      setErrorMsg('');
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    setStatus('uploading');
    setErrorMsg('');
    
    try {
      const formData = new FormData();
      formData.append('image', file);
      
      setStatus('processing');
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.details || data.error || 'Failed to process pipeline');
      }
      
      setModelUrl(data.meshUrl);
      setStatus('done');
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message);
      setStatus('error');
    } finally {
      setIsProcessing(false);
    }
  };

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
            
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/png, image/jpeg, image/tiff" 
              onChange={handleFileChange}
            />
            
            <div 
              className={`upload-zone ${file ? 'has-file' : ''}`} 
              onClick={handleFileClick}
              style={{ cursor: isProcessing ? 'not-allowed' : 'pointer', borderColor: file ? 'var(--accent-cyan)' : '' }}
            >
              <UploadCloud size={48} className="upload-icon" color={file ? 'var(--accent-cyan)' : 'var(--text-secondary)'} />
              <p style={{ fontWeight: 600 }}>{file ? file.name : 'Drop Satellite Image Here'}</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'JPEG, PNG, TIFF up to 100MB'}
              </p>
            </div>
            
            <div className="upload-zone" style={{ marginTop: '1rem', opacity: 0.5, cursor: 'not-allowed' }}>
              <Layers size={48} className="upload-icon" />
              <p style={{ fontWeight: 600 }}>Drop Reference DEM (Optional)</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                For absolute scale calibration
              </p>
            </div>

            <button 
              className="btn-primary" 
              style={{ marginTop: '1.5rem', opacity: (!file || isProcessing) ? 0.5 : 1, cursor: (!file || isProcessing) ? 'not-allowed' : 'pointer' }}
              onClick={handleProcess}
              disabled={!file || isProcessing}
            >
              {isProcessing ? (
                <><Loader size={16} className="spin" style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} /> Processing...</>
              ) : (
                <><Zap size={16} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} /> Process Pipeline</>
              )}
            </button>
            
            {errorMsg && (
              <div style={{ marginTop: '1rem', color: '#ff6b6b', fontSize: '0.85rem', padding: '0.5rem', background: 'rgba(255,107,107,0.1)', borderRadius: '4px' }}>
                <strong>Error:</strong> {errorMsg}
              </div>
            )}
          </div>

          <div className="glass-panel" style={{ padding: '1.5rem', flexGrow: 1 }}>
            <h3 style={{ marginBottom: '1.5rem', color: 'var(--text-primary)' }}>Pipeline Status</h3>
            <ul className="status-list">
              
              <li className="status-item">
                <div className={`status-icon ${status !== 'idle' ? 'done' : ''}`}>
                  {status !== 'idle' ? <CheckCircle size={14} /> : <span style={{ opacity: 0.3 }}>1</span>}
                </div>
                <div>
                  <strong style={{ color: status !== 'idle' ? 'var(--text-primary)' : 'var(--text-secondary)' }}>Stage 1: Image Upload</strong>
                  {status === 'uploading' && <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>Uploading image to server...</div>}
                  {(status === 'processing' || status === 'done') && <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Upload complete</div>}
                </div>
              </li>
              
              <li className="status-item">
                <div className={`status-icon ${status === 'processing' ? 'active' : (status === 'done' ? 'done' : '')}`}>
                  {status === 'done' ? <CheckCircle size={14} /> : (status === 'processing' ? <Loader size={14} className="spin" /> : <span style={{ opacity: 0.3 }}>2</span>)}
                </div>
                <div>
                  <strong style={{ color: status === 'processing' ? 'var(--accent-cyan)' : (status === 'done' ? 'var(--text-primary)' : 'var(--text-secondary)') }}>Stage 2: Depth Inference</strong>
                  {status === 'processing' && <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>Running Depth Anything V2...</div>}
                  {status === 'done' && <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Inference complete</div>}
                </div>
              </li>
              
              <li className="status-item">
                <div className={`status-icon ${status === 'processing' ? 'active' : (status === 'done' ? 'done' : '')}`}>
                  {status === 'done' ? <CheckCircle size={14} /> : (status === 'processing' ? <Loader size={14} className="spin" /> : <span style={{ opacity: 0.3 }}>3</span>)}
                </div>
                <div>
                  <strong style={{ color: status === 'processing' ? 'var(--accent-cyan)' : (status === 'done' ? 'var(--text-primary)' : 'var(--text-secondary)') }}>Stage 3: 3D Meshing</strong>
                  {status === 'processing' && <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>Generating 1M+ vertices...</div>}
                  {status === 'done' && <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Mesh generated successfully</div>}
                </div>
              </li>
              
            </ul>
          </div>
        </aside>

        <section className="canvas-container glass-panel">
          {/* Key forces component to unmount and remount when URL changes, breaking Three.js cache */}
          <Viewer3D key={modelUrl} modelUrl={modelUrl} />
          
          <div style={{ position: 'absolute', bottom: '20px', left: '20px', background: 'rgba(0,0,0,0.5)', padding: '10px 15px', borderRadius: '8px', backdropFilter: 'blur(8px)', border: '1px solid var(--border-glass)' }}>
            <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>3D Mesh Preview</h4>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Interactive Orbit View</p>
          </div>
        </section>
      </main>
    </div>
  );
}
