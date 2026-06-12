import { Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? 'navbar-scrolled' : ''}`}>
      <div className="navbar-inner container-wide">
        {location.pathname !== '/process' ? (
          <Link to="/" className="navbar-brand">
            <div className="navbar-logo">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                <defs>
                  <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
                    <stop offset="0%" stopColor="#4f8ef7" />
                    <stop offset="100%" stopColor="#7c3aed" />
                  </linearGradient>
                </defs>
                <circle cx="14" cy="14" r="12" stroke="url(#logo-grad)" strokeWidth="2.5" fill="none" />
                <circle cx="14" cy="14" r="5" fill="url(#logo-grad)" />
                <path d="M14 2 L14 6 M14 22 L14 26 M2 14 L6 14 M22 14 L26 14" stroke="url(#logo-grad)" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <span className="navbar-title">SPECTRA</span>
          </Link>
        ) : (
          <div className="navbar-brand-spacer" style={{ width: '150px' }}></div>
        )}

        <div className="navbar-links">
          <Link
            to="/"
            className={`navbar-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link
            to="/process"
            className={`navbar-link ${location.pathname === '/process' ? 'active' : ''}`}
          >
            Process
          </Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="navbar-link"
          >
            GitHub
          </a>
        </div>

        {location.pathname !== '/process' ? (
          <Link to="/process" className="btn btn-primary navbar-cta">
            Try Now
          </Link>
        ) : (
          <div className="navbar-cta-spacer" style={{ width: '110px' }}></div>
        )}
      </div>
    </nav>
  );
}
