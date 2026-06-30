import { useEffect, useCallback } from 'react';
import Navbar from '../components/Navbar';
import UploadZone from '../components/UploadZone';
import BeforeAfterSlider from '../components/BeforeAfterSlider';
import MetricsPanel from '../components/MetricsPanel';
import LoadingOverlay from '../components/LoadingOverlay';
import Strands from '../components/Strands';
import useColorize from '../hooks/useColorize';
import infraredImage from '../assets/image.png';
import realImage from '../assets/real.png';
import './ProcessPage.css';

export default function ProcessPage() {
  const {
    colorize,
    loadModel,
    reset,
    isLoading,
    result,
    error
  } = useColorize();

  // Load ONNX model on mount
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  const handleFileSelected = useCallback((file: File) => {
    colorize(file);
  }, [colorize]);

  const handleDownload = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = result.colorized_image;
    link.download = 'spectra_colorized.png';
    link.click();
  };

  const handleDownloadSR = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = result.super_resolved_image;
    link.download = 'spectra_super_resolved.png';
    link.click();
  };

  const handleTryAnother = () => {
    reset();
  };

  return (
    <>
      <Navbar />
      <LoadingOverlay isVisible={isLoading} />

      <main className="process-page">
        {/* Vertical branding removed per user request */}
        {/* Animated background across the whole layout */}
        <div className="process-bg">
          <div style={{ position: 'absolute', inset: 0, opacity: 0.5 }}>
            <Strands
              colors={['#2563eb', '#6366f1', '#0ea5e9', '#38bdf8']}
              speed={0.4}
              count={3}
              thickness={1.5}
              waviness={0.8}
              opacity={0.6}
            />
          </div>
        </div>

        <div className="container" style={{ position: 'relative' }}>

          <img
            src={infraredImage}
            alt="Infrared sample"
            className="floating-ir-image"
          />

          <img
            src={realImage}
            alt="Colorized sample"
            className="floating-rgb-image"
          />

          <div className="process-header animate-fade-in-down">
            <h1 className="process-title">
              Colorize Your <span className="process-title-accent">Infrared Image</span>
            </h1>
          </div>

          {/* 1x4 Full-Width Pipeline Layout */}
          <div className="process-pipeline-layout animate-fade-in-up">

            {/* Box 1: Input / Upload */}
            <div className="process-box glass-panel process-box-step">
              {!result ? (
                <div className="step-upload-container">
                  <UploadZone onFileSelected={handleFileSelected} disabled={isLoading} />
                  {error && (
                    <div className="process-error">
                      {error}
                    </div>
                  )}
                </div>
              ) : (
                <div className="step-image-container">
                  <div className="step-overlay-label">Input TIR (200m)</div>
                  <img src={result.input_image} alt="Input TIR" />
                </div>
              )}
            </div>

            {/* Box 2: Super-Resolved */}
            <div className="process-box glass-panel process-box-step">
              {!result ? (
                <div className="process-empty-state">
                  <div className="process-empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                      <polyline points="2 17 12 22 22 17"></polyline>
                      <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                  </div>
                  <h3>Super-Resolved TIR</h3>
                  <p>Awaiting Input<br />Enhances thermal resolution</p>
                </div>
              ) : (
                <div className="step-image-container">
                  <div className="step-overlay-label">Super-Resolved TIR</div>
                  <img src={result.super_resolved_image} alt="Super-Resolved TIR" />
                </div>
              )}
            </div>

            {/* Box 3: Colorized RGB */}
            <div className="process-box glass-panel process-box-step">
              {!result ? (
                <div className="process-empty-state">
                  <div className="process-empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="14.31" y1="8" x2="20.05" y2="17.94"></line>
                      <line x1="9.69" y1="8" x2="21.17" y2="8"></line>
                      <line x1="7.38" y1="12" x2="13.12" y2="2.06"></line>
                      <line x1="9.69" y1="16" x2="3.95" y2="6.06"></line>
                      <line x1="14.31" y1="16" x2="2.83" y2="16"></line>
                      <line x1="16.62" y1="12" x2="10.88" y2="21.94"></line>
                    </svg>
                  </div>
                  <h3>Colorized RGB</h3>
                  <p>Awaiting Inference<br />Translates TIR to visible RGB</p>
                </div>
              ) : (
                <div className="step-image-container">
                  <div className="step-overlay-label">Colorized RGB</div>
                  <img src={result.colorized_image} alt="Colorized RGB" />
                </div>
              )}
            </div>

            {/* Box 4: Ground Truth */}
            <div className="process-box glass-panel process-box-step">
              {!result ? (
                <div className="process-empty-state">
                  <div className="process-empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"></circle>
                      <circle cx="12" cy="12" r="6"></circle>
                      <circle cx="12" cy="12" r="2"></circle>
                    </svg>
                  </div>
                  <h3>Real RGB</h3>
                  <p>Optional Reference Image<br/>Upload for metric evaluation</p>
                </div>
              ) : (
                <div className="step-image-container">
                  <div className="step-overlay-label">Real RGB</div>
                  {result.reference_image ? (
                    <img src={result.reference_image} alt="Real RGB" />
                  ) : (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                      No reference image provided
                    </div>
                  )}
                </div>
              )}
            </div>

          </div>

          {/* Metrics and Actions displayed below when result exists */}
          {result && (
            <div className="process-footer animate-fade-in-up delay-1" style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '24px' }}>
              <MetricsPanel metrics={result.metrics} />
              <div className="process-actions">
                <button className="btn btn-primary" onClick={handleDownloadSR} id="download-sr-btn">
                  Download TIR (100m)
                </button>
                <button className="btn btn-primary" onClick={handleDownload} id="download-rgb-btn">
                  Download RGB (100m)
                </button>
                <button className="btn btn-secondary" onClick={handleTryAnother} id="try-another-btn">
                  Process Another
                </button>
              </div>
            </div>
          )}

          {/* Info cards horizontally centered below the split layout */}
          {!result && (
            <div className="process-info-grid animate-fade-in-up delay-2">
              <div className="process-info-card">
                <div className="process-info-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                  </svg>
                </div>
                <div className="process-info-text">
                  <h4>Supported Formats</h4>
                  <p>PNG, JPG, BMP, TIFF — up to 10MB</p>
                </div>
              </div>
              <div className="process-info-card">
                <div className="process-info-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                  </svg>
                </div>
                <div className="process-info-text">
                  <h4>Fast Processing</h4>
                  <p>Inference in under 2 seconds</p>
                </div>
              </div>
              <div className="process-info-card">
                <div className="process-info-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
                    <rect x="9" y="9" width="6" height="6"></rect>
                    <line x1="9" y1="1" x2="9" y2="4"></line>
                    <line x1="15" y1="1" x2="15" y2="4"></line>
                    <line x1="9" y1="20" x2="9" y2="23"></line>
                    <line x1="15" y1="20" x2="15" y2="23"></line>
                    <line x1="20" y1="9" x2="23" y2="9"></line>
                    <line x1="20" y1="14" x2="23" y2="14"></line>
                    <line x1="1" y1="9" x2="4" y2="9"></line>
                    <line x1="1" y1="14" x2="4" y2="14"></line>
                  </svg>
                </div>
                <div className="process-info-text">
                  <h4>Dual-Model Architecture</h4>
                  <p>ESRGAN for upscaling & Pix2Pix for colorization</p>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </>
  );
}
