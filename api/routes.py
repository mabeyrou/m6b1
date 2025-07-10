from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from tensorflow import keras
from loguru import logger
from os.path import join
from os import makedirs
from PIL import Image
import numpy as np
import base64
import io

from modules.models import predict
from schemas import DigitRequest
from database import get_db
from models import Digit

router = APIRouter()

model_path = join(".", "models", "cnn_latest.keras")
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
async def predict_digit(digitRequest: DigitRequest, db: Session = Depends(get_db)):
    img_bytes = base64.b64decode(digitRequest.image)
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_pil = img_pil.convert("L").resize((EXPECTED_DIMENSION, EXPECTED_DIMENSION))

    images_dir = "./data/images"
    makedirs(images_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = join(images_dir, f"image_{timestamp}.png")

    img_pil.save(image_path)
    logger.info(f"Image saved to {image_path}")

    img_array = (
        np.array(img_pil).reshape(1, EXPECTED_DIMENSION, EXPECTED_DIMENSION) / 255.0
    )

    try:
        logger.info("Starting prediction...")
        predictions = predict(model, img_array)
        prediction = int(np.argmax(predictions))
        confidence = predictions[prediction]
        logger.info(
            f"The model predicted: {prediction} with a confidence of {confidence}"
        )

        db_digit = Digit(
            img_path=image_path,
            predicted_digit=prediction,
            confidence=confidence,
            created_at=datetime.now(),
        )
        db.add(db_digit)
        db.commit()
        db.refresh(db_digit)

        return {"success": True, "prediction": prediction}
    except Exception as err:
        logger.error(f"An error occured during prediction: {err}")
        detail_message = f"Something went wrong during prediction: {err}"
        raise HTTPException(status_code=500, detail=detail_message)
