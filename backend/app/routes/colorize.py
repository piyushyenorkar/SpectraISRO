"""
SPECTRA — Process API Route
POST /process — accepts IR image, runs Model A + Model B, returns 4 images + metrics
"""

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.inference import inference_service
from app.utils.preprocessing import validate_image

router = APIRouter()


@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Process an infrared image through the 2-stage pipeline.

    Accepts: multipart/form-data with 'file' field (PNG/JPG)
    Returns: JSON with 4 images (base64) and metrics
    """
    # Read file
    file_bytes = await file.read()

    # Validate
    is_valid, message = validate_image(file_bytes)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    # Check content type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PNG, JPG, BMP, or TIFF.",
        )

    try:
        result = inference_service.process(file_bytes)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
