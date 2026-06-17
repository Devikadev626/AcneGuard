import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException

from src.predict import predict_image
from database.database import create_prediction, init_db

app = FastAPI(
    title="AcneGuard API"
)

@app.on_event("startup")
def on_startup():
    # Automatically initialize database on startup if not already initialized
    init_db()

@app.get("/")
def home():

    return {
        "message":
        "AcneGuard API Running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile
):

    file_path = (
        f"temp_{file.filename}"
    )

    try:
        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {str(e)}"
        )

    try:
        try:
            severity, confidence = predict_image(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

        try:
            prediction_id = create_prediction(
                image_name=file.filename,
                severity=severity,
                confidence=confidence
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database storage failed: {str(e)}"
            )
    finally:
        # Cleanup temporary image file after inference and database entry
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "prediction_id": prediction_id,
        "severity": severity,
        "confidence": confidence
    }