import './HowItWorks.css';

const steps = [
  {
    number: '01',
    title: 'Upload',
    description: 'Drop your raw infrared or thermal grayscale image. Supports PNG, JPG, BMP, TIFF.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
  {
    number: '02',
    title: 'Process',
    description: 'Our pix2pix GAN analyzes thermal gradients, reconstructs color channels, and enhances details — all in your browser via ONNX Runtime.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
  {
    number: '03',
    title: 'Compare',
    description: 'Instantly compare the original IR and colorized output with our interactive before/after slider. Download the enhanced image as PNG.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <line x1="12" y1="3" x2="12" y2="21" />
        <path d="M3 12h18" />
      </svg>
    ),
  },
];

export default function HowItWorks() {
  return (
    <section className="how-it-works section" id="how-it-works">
      <div className="hiw-bg-glow"></div>
      <div className="container" style={{ position: 'relative', zIndex: 10 }}>
        <h2 className="section-title" style={{ textAlign: 'center' }}>How It Works</h2>
        <p className="section-subtitle" style={{ textAlign: 'center', maxWidth: '600px', margin: '0 auto 64px' }}>
          Three simple steps from raw thermal data to a vivid, interpretable image.
        </p>

        <div className="hiw-steps">
          {steps.map((step, i) => (
            <div key={step.number} className={`hiw-step-wrapper animate-fade-in-up delay-${i + 1}`}>
              <div className="hiw-step glass-panel">
                <div className="hiw-step-bg-number">{step.number}</div>
                <div className="hiw-step-icon-wrapper">
                  <div className="hiw-step-icon">{step.icon}</div>
                </div>
                <h3 className="hiw-step-title">{step.title}</h3>
                <p className="hiw-step-desc">{step.description}</p>
              </div>
              {i < steps.length - 1 && (
                <div className="hiw-connector-animated">
                  <div className="hiw-connector-line"></div>
                  <div className="hiw-connector-dot"></div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
