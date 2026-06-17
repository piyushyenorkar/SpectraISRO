"""
SPECTRA — FastAPI Backend Application
Serves the pix2pix GAN model for IR image colorization.

Run with: uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.inference import inference_service
from app.routes.colorize import router as colorize_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    print("🔬 SPECTRA Backend starting...")
    inference_service.load_model()
    print("🚀 Ready to colorize!")
    yield
    print("👋 SPECTRA Backend shutting down.")


app = FastAPI(
    title="SPECTRA API",
    description="Infrared Image Colorization & Enhancement API powered by pix2pix GAN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(colorize_router, prefix="/api", tags=["Colorization"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_service.model is not None,
        "device": str(inference_service.device) if inference_service.device else "not initialized",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SPECTRA API",
        "version": "1.0.0",
        "description": "Infrared Image Colorization & Enhancement",
        "endpoints": {
            "POST /api/process": "Upload IR image for 2-stage processing",
            "GET /health": "API health check",
        },
    }
