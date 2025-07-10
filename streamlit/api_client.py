from os import getenv
from dotenv import load_dotenv
import requests
from loguru import logger

load_dotenv()

API_URL = getenv("API_URL")


def predict(image):
    try:
        response = requests.post(
            url=f"{API_URL}/predict", json={"image": image}, timeout=300
        )
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as error:
        logger.error(f"Error while predicting: {error}")
        return {"success": False, "message": "Something went wrong while predicting"}
