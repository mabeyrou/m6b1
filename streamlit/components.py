import streamlit as st
from streamlit_drawable_canvas import st_canvas
from loguru import logger
from PIL import Image
import numpy as np
import base64
import io

from api_client import predict


logger.remove()
logger.add(
    "./logs/dev_streamlit.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="TRACE",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

CANVAS_DIMENSION = 192


def main():
    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            stroke_width=15,
            stroke_color="#000",
            background_color="#eee",
            height=CANVAS_DIMENSION,
            width=CANVAS_DIMENSION,
            drawing_mode="freedraw",
            key="canvas",
        )
        feedback_section()

    with col2:
        is_empty = not canvas_result.json_data["objects"]
        has_image_data = canvas_result.image_data is not None

        predict_button = st.button(type="primary", label="Predict", disabled=is_empty)
        if "prediction_value" not in st.session_state:
            st.session_state.prediction_value = "?"
            st.session_state.error_message = ""

        if has_image_data and not is_empty and predict_button:
            img_array = canvas_result.image_data.astype(np.uint8)
            img_pil = Image.fromarray(img_array)

            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            prediction = predict(img_base64)
            logger.info(f"The model predicted a {prediction}")

            if prediction["success"]:
                st.session_state.prediction_value = prediction["prediction"]
                st.session_state.error_message = ""
            else:
                st.session_state.prediction_value = "?"
                st.session_state.error_message = (
                    "Une erreur est survenue lors de la prédiction."
                )

        st.metric(label="Prédiction", value=st.session_state.prediction_value)
        if st.session_state.error_message:
            st.error(st.session_state.error_message)


def feedback_section():
    bcol1, bcol2 = st.columns(2)
    correct_clicked = bcol1.button("Correct", key="btn_correct", icon="✅")
    error_clicked = bcol2.button("Error", key="btn_error", icon="❌")

    # Affichage selon le bouton cliqué
    if "show_numpad" not in st.session_state:
        st.session_state.show_numpad = False
    if "show_check" not in st.session_state:
        st.session_state.show_check = False

    if correct_clicked:
        st.session_state.show_check = True
        st.session_state.show_numpad = False
    if error_clicked:
        st.session_state.show_numpad = True
        st.session_state.show_check = False

    if st.session_state.show_check:
        st.markdown(
            "<div style='font-size:3em; color:green; text-align:center;'>✅</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.show_numpad:
        numpad()


def numpad():
    # Initialisation de la valeur si besoin
    if "last_numpad_value" not in st.session_state:
        st.session_state.last_numpad_value = ""

    # Affichage de la dernière valeur cliquée et bouton pour vider
    st.markdown(
        f"<div style='font-size:2em; text-align:center;'>Dernier chiffre : <b>{st.session_state.last_numpad_value or '-'}</b></div>",
        unsafe_allow_html=True,
    )
    if st.button("Vider", key="numpad_clear", use_container_width=True):
        st.session_state.last_numpad_value = ""

    # Numpad
    cols = st.columns(3)
    for i, n in enumerate([7, 8, 9]):
        if cols[i].button(str(n), key=f"numpad_{n}", use_container_width=True):
            st.session_state.last_numpad_value = str(n)

    cols = st.columns(3)
    for i, n in enumerate([4, 5, 6]):
        if cols[i].button(str(n), key=f"numpad_{n}", use_container_width=True):
            st.session_state.last_numpad_value = str(n)

    cols = st.columns(3)
    for i, n in enumerate([1, 2, 3]):
        if cols[i].button(str(n), key=f"numpad_{n}", use_container_width=True):
            st.session_state.last_numpad_value = str(n)

    cols = st.columns([1, 2])
    if cols[0].button("0", key="numpad_0", use_container_width=True):
        st.session_state.last_numpad_value = "0"
    if cols[1].button("Fix", key="numpad_fix", type="primary", use_container_width=True):
        # Ici, tu peux envoyer la valeur à l'endroit voulu
        logger.info(f"Valeur envoyée via Fix : {st.session_state.last_numpad_value}")
        st.success(f"Valeur envoyée : {st.session_state.last_numpad_value}")


if __name__ == "__main__":
    st.set_page_config(page_title="Digit Prediction App", page_icon=":pencil2:")
    st.title("✏️ Digit Prediction App")
    main()
