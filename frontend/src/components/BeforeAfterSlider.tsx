import { useState, useRef, useCallback, useEffect } from 'react';
import './BeforeAfterSlider.css';

interface BeforeAfterSliderProps {
  beforeImage: string;  // IR image (base64 data URI)
  afterImage: string;   // Colorized image (base64 data URI)
  beforeLabel?: string;
  afterLabel?: string;
}

export default function BeforeAfterSlider({
  beforeImage,
  afterImage,
  beforeLabel = 'INFRARED',
  afterLabel = 'COLORIZED',
}: BeforeAfterSliderProps) {
  const [position, setPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updatePosition = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(percent);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    updatePosition(e.clientX);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    setIsDragging(true);
    updatePosition(e.touches[0].clientX);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) updatePosition(e.clientX);
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging) updatePosition(e.touches[0].clientX);
    };

    const handleEnd = () => setIsDragging(false);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleEnd);
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    window.addEventListener('touchend', handleEnd);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleEnd);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging, updatePosition]);

  return (
    <div
      ref={containerRef}
      className={`slider-container ${isDragging ? 'slider-dragging' : ''}`}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      id="before-after-slider"
    >
      {/* After image (full width, visible on right side) */}
      <div className="slider-after">
        <img src={afterImage} alt="Colorized output" draggable={false} />
        <span className="slider-label slider-label-after">{afterLabel}</span>
      </div>

      {/* Before image (clipped to position) */}
      <div
        className="slider-before"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img src={beforeImage} alt="IR input" draggable={false} />
        <span className="slider-label slider-label-before">{beforeLabel}</span>
      </div>

      {/* Divider line + handle */}
      <div className="slider-divider" style={{ left: `${position}%` }}>
        <div className="slider-line" />
        <div className="slider-handle">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </div>
      </div>
    </div>
  );
}
