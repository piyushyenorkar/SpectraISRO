import { useState, useEffect } from 'react';
import './LoadingOverlay.css';

const processingSteps = [
  'Analyzing thermal patterns...',
  'Extracting feature maps...',
  'Reconstructing color channels...',
  'Applying spectral enhancement...',
  'Generating final output...',
];

interface LoadingOverlayProps {
  isVisible: boolean;
}

export default function LoadingOverlay({ isVisible }: LoadingOverlayProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (!isVisible) {
      setStepIndex(0);
      setElapsedMs(0);
      return;
    }

    const start = Date.now();
    const timerInterval = setInterval(() => {
      setElapsedMs(Date.now() - start);
    }, 50);

    const interval = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % processingSteps.length);
    }, 1500);

    return () => {
      clearInterval(interval);
      clearInterval(timerInterval);
    };
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="loading-overlay" id="loading-overlay">
      <div className="loading-content">
        {/* Neural network animation */}
        <div className="loading-visual">
          <div className="loading-ring loading-ring-1" />
          <div className="loading-ring loading-ring-2" />
          <div className="loading-ring loading-ring-3" />
          <div className="loading-core">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
            </svg>
          </div>
        </div>

        <h3 className="loading-title">Processing with Two-Stage Pipeline</h3>
        <p className="loading-step" key={stepIndex}>
          {processingSteps[stepIndex]}
        </p>

        <div style={{ marginTop: '16px', fontSize: '18px', fontFamily: 'monospace', color: 'var(--brand-blue)' }}>
          {(elapsedMs / 1000).toFixed(2)}s elapsed
        </div>

        {/* Progress dots */}
        <div className="loading-dots">
          {processingSteps.map((_, i) => (
            <div
              key={i}
              className={`loading-dot ${i === stepIndex ? 'loading-dot-active' : ''} ${i < stepIndex ? 'loading-dot-done' : ''}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
