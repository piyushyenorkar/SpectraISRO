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
              <img src="/faviconspectra.png" alt="SPECTRA Logo" width="28" height="28" />
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
