import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-brand">
            <div className="footer-logo">
              <svg width="24" height="24" viewBox="0 0 28 28" fill="none">
                <defs>
                  <linearGradient id="footer-logo-grad" x1="0" y1="0" x2="28" y2="28">
                    <stop offset="0%" stopColor="#4f8ef7" />
                    <stop offset="100%" stopColor="#7c3aed" />
                  </linearGradient>
                </defs>
                <circle cx="14" cy="14" r="12" stroke="url(#footer-logo-grad)" strokeWidth="2.5" fill="none" />
                <circle cx="14" cy="14" r="5" fill="url(#footer-logo-grad)" />
              </svg>
              <span>SPECTRA</span>
            </div>
            <p className="footer-tagline">
              AI-Powered Infrared Image Colorization & Enhancement
            </p>
          </div>

          <div className="footer-boxes">
            <div className="footer-box footer-section">
              <h4>Technology</h4>
              <div className="footer-tech-tags">
                <span className="footer-tag">PyTorch</span>
                <span className="footer-tag">pix2pix GAN</span>
                <span className="footer-tag">ONNX Runtime</span>
                <span className="footer-tag">React</span>
                <span className="footer-tag">FastAPI</span>
              </div>
            </div>

            <div className="footer-box footer-section">
              <h4>Hackathon</h4>
              <div className="footer-hackathon-info">
                <p className="footer-text">Bharatiya Antariksh Hackathon 2026</p>
                <p className="footer-text">Problem Statement 10</p>
                <p className="footer-text">ISRO × Hack2skill</p>
              </div>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p>© 2026 Team SPECTRA — Built for BAH 2026</p>
          <p className="footer-isro">
            Indian Space Research Organisation
          </p>
        </div>
      </div>
    </footer>
  );
}
