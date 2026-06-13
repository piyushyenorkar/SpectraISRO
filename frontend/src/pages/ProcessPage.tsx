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

  const handleTryAnother = () => {
    reset();
  };

  return (
    <>
      <Navbar />
      <LoadingOverlay isVisible={isLoading} />

      <main className="process-page">
        <div className="vertical-branding-container">
          <img src="/faviconspectra.png" alt="SPECTRA Logo" className="vertical-branding-logo" />
          <div className="vertical-branding">SPECTRA</div>
        </div>
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

          {/* Split Layout */}
          <div className="process-split-layout animate-fade-in-up">
            <div className="process-center-logo">
              <img src="/faviconspectra.png" alt="SPECTRA Logo" />
            </div>

            {/* Left Box: Upload */}
            <div className="process-box glass-panel process-box-upload">
              <UploadZone onFileSelected={handleFileSelected} disabled={isLoading} />

              {error && (
                <div className="process-error">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  {error}
                </div>
              )}
            </div>

            {/* Right Box: Results or Placeholder */}
            <div className="process-box glass-panel process-box-results">
              {!result ? (
                <div className="process-empty-state">
                  <div className="process-empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  </div>
                  <h3>Waiting for Image</h3>
                  <p>Upload an IR image on the left to see the RGB transformation here.</p>
                </div>
              ) : (
                <div className="process-results-content">
                  <div className="process-slider-wrapper">
                    <BeforeAfterSlider
                      beforeImage={result.ir_preview}
                      afterImage={result.colorized_image}
                    />
                  </div>

                  <MetricsPanel metrics={result.metrics} />

                  <div className="process-actions">
                    <button className="btn btn-primary" onClick={handleDownload} id="download-btn">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                      </svg>
                      Download
                    </button>
                    <button className="btn btn-secondary" onClick={handleTryAnother} id="try-another-btn">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                        <polyline points="1 4 1 10 7 10" />
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                      </svg>
                      Try Another
                    </button>
                  </div>
                </div>
              )}
            </div>

          </div>

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
                  <h4>pix2pix GAN</h4>
                  <p>U-Net generator with PatchGAN discriminator</p>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </>
  );
}
