import { useEffect, useState } from 'react';
import './MetricsPanel.css';

interface MetricsPanelProps {
  metrics: {
    inference_time_ms: number;
    psnr: number;
    ssim: number;
    fid: number;
  };
}

function AnimatedNumber({ value, suffix = '', decimals = 1 }: { value: number; suffix?: string; decimals?: number }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    let start = 0;
    const duration = 800;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      start = value * eased;
      setDisplay(start);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }, [value]);

  return <>{display.toFixed(decimals)}{suffix}</>;
}

export default function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <div className="metrics-panel" id="metrics-panel">
      <h3 className="metrics-title">Enhancement Metrics</h3>

      <div className="metrics-grid">
        {/* Inference Time */}
        <div className="metric-card metric-card-purple">
          <div className="metric-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          </div>
          <div className="metric-info">
            <span className="metric-value">
              <AnimatedNumber value={metrics.inference_time_ms} suffix="ms" decimals={0} />
            </span>
            <span className="metric-label">Inference Time</span>
          </div>
        </div>

        {/* PSNR */}
        <div className="metric-card metric-card-blue">
          <div className="metric-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M12 2v20M2 12h20" />
              <path d="M17 7l-5 5-5-5" />
            </svg>
          </div>
          <div className="metric-info">
            <span className="metric-value">
              <AnimatedNumber value={metrics.psnr} suffix=" dB" decimals={2} />
            </span>
            <span className="metric-label">PSNR (Quality)</span>
          </div>
        </div>

        {/* SSIM */}
        <div className="metric-card metric-card-green">
          <div className="metric-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
              <polyline points="17 6 23 6 23 12" />
            </svg>
          </div>
          <div className="metric-info">
            <span className="metric-value">
              <AnimatedNumber value={metrics.ssim} decimals={3} />
            </span>
            <span className="metric-label">SSIM (Structure)</span>
          </div>
        </div>

        {/* FID */}
        <div className="metric-card metric-card-accent">
          <div className="metric-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
            </svg>
          </div>
          <div className="metric-info">
            <span className="metric-value">
              <AnimatedNumber value={metrics.fid} decimals={2} />
            </span>
            <span className="metric-label">FID Score</span>
          </div>
        </div>
      </div>
    </div>
  );
}
