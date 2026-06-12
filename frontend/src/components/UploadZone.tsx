import { useState, useRef, useCallback } from 'react';
import './UploadZone.css';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function UploadZone({ onFileSelected, disabled = false }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
  const MAX_SIZE_MB = 10;

  const validateFile = (file: File): boolean => {
    setError('');

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(`Unsupported format: ${file.type}. Use PNG, JPG, BMP, or TIFF.`);
      return false;
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > MAX_SIZE_MB) {
      setError(`File too large: ${sizeMB.toFixed(1)}MB (max ${MAX_SIZE_MB}MB)`);
      return false;
    }

    return true;
  };

  const handleFile = useCallback((file: File) => {
    if (!validateFile(file)) return;

    setFileName(file.name);

    // Generate preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    onFileSelected(file);
  }, [onFileSelected]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    if (!disabled) fileInputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const resetUpload = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setFileName('');
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div
      className={`upload-zone ${isDragging ? 'upload-zone-dragging' : ''} ${preview ? 'upload-zone-has-preview' : ''} ${disabled ? 'upload-zone-disabled' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      id="upload-zone"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif"
        onChange={handleInputChange}
        className="upload-input"
      />

      {preview ? (
        <div className="upload-preview">
          <img src={preview} alt="IR preview" className="upload-preview-img" />
          <div className="upload-preview-overlay">
            <span className="upload-preview-name">{fileName}</span>
            <button className="upload-preview-reset" onClick={resetUpload}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              Change
            </button>
          </div>
        </div>
      ) : (
        <div className="upload-placeholder">
          <div className="upload-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <h3 className="upload-title">
            {isDragging ? 'Drop your image here' : 'Upload Infrared Image'}
          </h3>
          <p className="upload-hint">
            Drag & drop or click to select<br/>
            PNG, JPG, BMP, TIFF · Max 10MB
          </p>
        </div>
      )}

      {error && (
        <div className="upload-error">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          {error}
        </div>
      )}
    </div>
  );
}
