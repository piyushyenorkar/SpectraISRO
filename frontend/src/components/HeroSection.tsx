import { Link } from 'react-router-dom';
import './HeroSection.css';
import Strands from './Strands';

export default function HeroSection() {
  return (
    <section className="hero">
      {/* Animated background */}
      <div className="hero-bg">
        <div style={{ position: 'absolute', inset: 0, opacity: 0.6 }}>
          <Strands 
            colors={['#2563eb', '#6366f1', '#0ea5e9', '#38bdf8']} 
            speed={0.6}
            count={4}
            thickness={1}
            waviness={1.2}
            opacity={0.7}
          />
        </div>
        <div className="hero-orb hero-orb-1" />
        <div className="hero-orb hero-orb-2" />
        <div className="hero-orb hero-orb-3" />
        <div className="hero-grid" />
      </div>

      <div className="hero-content container">
        <div className="hero-badge animate-fade-in">
          <span className="hero-badge-dot" />
          BAH 2026 — PS 10 · ISRO × Hack2skill
        </div>

        <h1 className="hero-title animate-fade-in-up delay-1">
          Transform
          <span className="hero-title-gradient"> Infrared </span>
          Into Vision
        </h1>

        <p className="hero-subtitle animate-fade-in-up delay-2">
          AI-powered colorization that converts raw thermal infrared imagery into
          natural, interpretable RGB images — in real-time, right in your browser.
        </p>

        <div className="hero-actions animate-fade-in-up delay-3">
          <Link to="/process" className="btn btn-primary btn-lg hero-cta">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload IR Image
          </Link>
          <a href="#how-it-works" className="btn btn-secondary btn-lg">
            How It Works
          </a>
        </div>

        <div className="hero-stats animate-fade-in-up delay-4">
          <div className="hero-stat">
            <span className="hero-stat-value">pix2pix</span>
            <span className="hero-stat-label">GAN Architecture</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">&lt; 2s</span>
            <span className="hero-stat-label">Inference Time</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">256×256</span>
            <span className="hero-stat-label">Resolution</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">Browser</span>
            <span className="hero-stat-label">ONNX Runtime</span>
          </div>
        </div>
      </div>
    </section>
  );
}
