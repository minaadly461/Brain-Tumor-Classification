
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from src.predict import BrainTumorPredictor


# Initialize API
app = FastAPI(
    title="Brain Tumor Classification API",
    description="MRI classification using fine-tuned ResNet50",
    version="1.0.0"
)


# Load model once when the API starts
predictor = BrainTumorPredictor()


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Classification API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "ResNet50 Fine-tuned"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    file_extension = Path(file.filename or "").suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are supported."
        )

    try:
        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Validate image content
        image = Image.open(BytesIO(contents))
        image.verify()

        # Reopen image after verification
        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )

    try:
        # Run prediction using existing pipeline
        result = predictor.predict(image)

        return {
            "filename": file.filename,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
