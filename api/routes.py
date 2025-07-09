from fastapi import APIRouter, HTTPException
from loguru import logger
from os.path import join
from tensorflow import keras
import base64
import io
from PIL import Image
import numpy as np


from modules.models import predict
from schemas import DigitRequest

router = APIRouter()

model_path = join("models", "cnn_latest.keras")
try:
    logger.debug(model_path)
    model = keras.models.load_model(model_path)
except Exception as error:
    logger.debug(model_path)
    logger.error(f"Error loading model from {model_path}: {error}")


EXPECTED_DIMENSION = 28


@router.get("/")
async def home():
    return {"message": "The server is up and running!"}


@router.get("/health")
async def heath():
    return {"status": "ok"}


@router.post("/predict")
async def predict_digit(digitRequest: DigitRequest):
    img_bytes = base64.b64decode(digitRequest.image)
    img_pil = (
        Image.open(io.BytesIO(img_bytes))
        .convert("L")
        .resize((EXPECTED_DIMENSION, EXPECTED_DIMENSION))
    )
    img_array = (
        np.array(img_pil).reshape(1, EXPECTED_DIMENSION, EXPECTED_DIMENSION) / 255.0
    )

    try:
        logger.info("Starting prediction...")
        predictions = predict(model, img_array)
        prediction = int(np.argmax(predictions))
        logger.info(f"The model predicted: {prediction}")

        return {"success": True, "prediction": prediction}
    except Exception as err:
        logger.error(f"An error occured during prediction: {err}")
        detail_message = f"Something went wrong during prediction: {err}"
        raise HTTPException(status_code=500, detail=detail_message)
