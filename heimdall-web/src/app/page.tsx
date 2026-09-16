"use client";

import React, { useState, useRef } from 'react';
import { UploadCloud, Layers, Zap, Hexagon, CheckCircle, Loader, Download, FileText } from 'lucide-react';
import Viewer3D from '@/components/Viewer3D';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [refDem, setRefDem] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<string>('idle');
  const [modelUrl, setModelUrl] = useState<string>('/models/stadium_mesh.glb');
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [outputFiles, setOutputFiles] = useState<{depthPng?: string; geotiff?: string; mesh?: string}>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const demInputRef = useRef<HTMLInputElement>(null);

  const handleFileClick = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleDemClick = () => {
    if (!isProcessing) {
      demInputRef.current?.click();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus('idle');
      setErrorMsg('');
    }
  };

  const handleDemChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setRefDem(e.target.files[0]);
    }
  };

  const [metrics, setMetrics] = useState<{min?: number, max?: number, mean?: number}>({});

  const handleProcess = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    setStatus('uploading');
    setErrorMsg('');
    setOutputFiles({});
    setMetrics({});
    
    try {
      const formData = new FormData();
      formData.append('image', file);
      if (refDem) {
        formData.append('reference_dem', refDem);
      }
      
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to process pipeline');
      }

      if (!response.body) throw new Error('No response body');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep the incomplete part in the buffer
        
        for (const chunk of lines) {
          const lines = chunk.split('\n');
          let eventType = 'message';
          let eventData = '';
          
          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.substring(7);
            if (line.startsWith('data: ')) eventData = line.substring(6);
          }
          
          if (eventData) {
            const parsed = JSON.parse(eventData);
            if (eventType === 'status') {
              setStatus('processing');
              // UI will just show general processing state, but we could add a sub-status label
            } else if (eventType === 'metric') {
              setMetrics(prev => ({ ...prev, [parsed.key]: parsed.value }));
            } else if (eventType === 'done') {
              setModelUrl(parsed.meshUrl);
              setOutputFiles({
                depthPng: parsed.depthPng,
                geotiff: parsed.geotiff,
                mesh: parsed.meshUrl,
              });
              setStatus('done');
            } else if (eventType === 'error') {
              throw new Error(parsed.message);
            }
          }
        }
      }
      
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
        <div className="header-subtitle">
          Single-View Height Estimation &amp; 3D Flythrough
        </div>
      </header>

      <main className="main-layout">
        <aside className="dashboard-sidebar">
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 className="panel-heading">New Processing Job</h3>
            
            {/* Hidden file inputs */}
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/png, image/jpeg, image/tiff" 
              onChange={handleFileChange}
            />
            <input
              type="file"
              ref={demInputRef}
              style={{ display: 'none' }}
              accept=".tif,.tiff"
              onChange={handleDemChange}
            />
            
            {/* Satellite Image Upload */}
            <div 
              className={`upload-zone ${file ? 'has-file' : ''}`} 
              onClick={handleFileClick}
              style={{ 
                cursor: isProcessing ? 'not-allowed' : 'pointer', 
                borderColor: file ? 'var(--accent-cyan)' : '' 
              }}
            >
              <UploadCloud 
                size={48} 
                className="upload-icon" 
                color={file ? 'var(--accent-cyan)' : 'var(--text-secondary)'} 
              />
              <p className="upload-title">{file ? file.name : 'Drop Satellite Image Here'}</p>
              <p className="upload-meta">
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'JPEG, PNG, TIFF up to 100MB'}
              </p>
            </div>
            
            {/* Reference DEM Upload - NOW ENABLED */}
            <div 
              className={`upload-zone ${refDem ? 'has-file' : ''}`}
              onClick={handleDemClick}
              style={{ 
                marginTop: '1rem', 
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                borderColor: refDem ? 'var(--accent-purple)' : '',
              }}
            >
              <Layers 
                size={48} 
                className="upload-icon" 
                color={refDem ? 'var(--accent-purple)' : 'var(--text-secondary)'}
              />
              <p className="upload-title">
                {refDem ? refDem.name : 'Drop Reference DEM (Optional)'}
              </p>
              <p className="upload-meta">
                {refDem 
                  ? `${(refDem.size / 1024 / 1024).toFixed(2)} MB — For absolute scale calibration`
                  : 'GeoTIFF for absolute scale calibration (SRTM, ASTER, etc.)'
                }
              </p>
            </div>

            <button 
              className="btn-primary" 
              style={{ 
                marginTop: '1.5rem', 
                opacity: (!file || isProcessing) ? 0.5 : 1, 
                cursor: (!file || isProcessing) ? 'not-allowed' : 'pointer' 
              }}
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
              <div className="error-box">
                <strong>Error:</strong> {errorMsg}
              </div>
            )}
          </div>

          <div className="glass-panel" style={{ padding: '1.5rem', flexGrow: 1 }}>
            <h3 className="panel-heading">Pipeline Status</h3>
            <ul className="status-list">
              
              <li className="status-item">
                <div className={`status-icon ${status !== 'idle' ? 'done' : ''}`}>
                  {status !== 'idle' ? <CheckCircle size={14} /> : <span className="status-num">1</span>}
                </div>
                <div>
                  <strong className={`status-label ${status !== 'idle' ? 'active' : ''}`}>Stage 1: Image Upload</strong>
                  {status === 'uploading' && <div className="status-detail status-detail--active">Uploading image to server...</div>}
                  {(status === 'processing' || status === 'done') && <div className="status-detail">Upload complete</div>}
                </div>
              </li>
              
              <li className="status-item">
                <div className={`status-icon ${status === 'processing' ? 'active' : (status === 'done' ? 'done' : '')}`}>
                  {status === 'done' ? <CheckCircle size={14} /> : (status === 'processing' ? <Loader size={14} className="spin" /> : <span className="status-num">2</span>)}
                </div>
                <div>
                  <strong className={`status-label ${status === 'processing' || status === 'done' ? 'active' : ''}`}>Stage 2: Depth Inference</strong>
                  {status === 'processing' && <div className="status-detail status-detail--active">Running Depth Anything V2...</div>}
                  {status === 'done' && <div className="status-detail">Inference complete</div>}
                </div>
              </li>
              
              <li className="status-item">
                <div className={`status-icon ${status === 'processing' ? 'active' : (status === 'done' ? 'done' : '')}`}>
                  {status === 'done' ? <CheckCircle size={14} /> : (status === 'processing' ? <Loader size={14} className="spin" /> : <span className="status-num">3</span>)}
                </div>
                <div>
                  <strong className={`status-label ${status === 'processing' || status === 'done' ? 'active' : ''}`}>Stage 3: 3D Meshing</strong>
                  {status === 'processing' && <div className="status-detail status-detail--active">Generating mesh vertices...</div>}
                  {status === 'done' && <div className="status-detail">Mesh generated successfully</div>}
                </div>
              </li>
              
            </ul>

            {/* Download links after processing */}
            {status === 'done' && (
              <div className="downloads-section">
                <h4 className="downloads-heading">Downloads</h4>
                {outputFiles.depthPng && (
                  <a href={outputFiles.depthPng} download className="download-link">
                    <Download size={14} /> Depth Map (PNG)
                  </a>
                )}
                {outputFiles.geotiff && (
                  <a href={outputFiles.geotiff} download className="download-link">
                    <FileText size={14} /> DSM GeoTIFF
                  </a>
                )}
                {outputFiles.mesh && (
                  <a href={outputFiles.mesh} download className="download-link">
                    <Download size={14} /> 3D Mesh (PLY)
                  </a>
                )}
              </div>
            )}
          </div>

          {Object.keys(metrics).length > 0 && (
            <div className="glass-panel" style={{ padding: '1.5rem', flexGrow: 1 }}>
              <h3 className="panel-heading">Metrics Dashboard</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', textAlign: 'center' }}>
                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Min Height</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{metrics.min?.toFixed(2) || '---'} m</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Max Height</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{metrics.max?.toFixed(2) || '---'} m</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Mean Height</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{metrics.mean?.toFixed(2) || '---'} m</div>
                </div>
              </div>
            </div>
          )}
        </aside>

        <section className="canvas-container glass-panel">
          <Viewer3D key={modelUrl} modelUrl={modelUrl} />
          
          <div className="viewer-label">
            <h4 className="viewer-label-title">3D Mesh Preview</h4>
            <p className="viewer-label-sub">Interactive Orbit View</p>
          </div>
        </section>
      </main>
    </div>
  );
}
