import './SampleGallery.css';

// These will be replaced with real samples after training
const samples = [
  {
    id: 1,
    label: 'Urban Street Scene',
    description: 'Pedestrians and vehicles captured in thermal',
  },
  {
    id: 2,
    label: 'Night Surveillance',
    description: 'Low-light thermal imaging enhanced with color',
  },
  {
    id: 3,
    label: 'Building Thermal Map',
    description: 'Heat loss detection and structural analysis',
  },
];

export default function SampleGallery() {
  return (
    <section className="gallery section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Sample Results</h2>
          <p className="section-subtitle">
            See what SPECTRA can do — real outputs from our trained pix2pix GAN model.
          </p>
        </div>

        <div className="gallery-grid">
          {samples.map((sample, i) => (
            <div key={sample.id} className={`gallery-card animate-fade-in-up delay-${i + 1}`}>
              <div className="gallery-preview">
                <div className="gallery-side gallery-ir">
                  <div className="gallery-placeholder-ir">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <path d="M21 15l-5-5L5 21" />
                    </svg>
                    <span>IR Input</span>
                  </div>
                </div>
                <div className="gallery-side gallery-rgb">
                  <div className="gallery-placeholder-rgb">
                    <div className="scanner-line scanner-line-rgb"></div>
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <path d="M21 15l-5-5L5 21" />
                    </svg>
                    <span>Colorized</span>
                  </div>
                </div>
                <div className="gallery-hover-indicator">
                  <span>Hover to colorize</span>
                </div>
              </div>
              <div className="gallery-info">
                <h3 className="gallery-label">{sample.label}</h3>
                <p className="gallery-desc">{sample.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
