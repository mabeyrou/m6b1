from os import getenv
from dotenv import load_dotenv
import requests
from loguru import logger
import streamlit as st

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
        st.error(f"Something went wrong while predicting: {str(error)}")
        return None


def feedback(feedback_request):
    try:
        response = requests.post(
            url=f"{API_URL}/feedback", json=feedback_request, timeout=5
        )
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as error:
        logger.error(f"Error during feedback: {error}")
        st.error(f"Something went wrong during : {str(error)}")
        return None
