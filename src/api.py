from fastapi import FastAPI
from fastapi import UploadFile
import shutil

from src.predict import predict_image

app = FastAPI(
    title="AcneGuard API"
)


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

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    severity, confidence = (
        predict_image(
            file_path
        )
    )

    return {

        "severity":
        severity,

        "confidence":
        confidence
    }